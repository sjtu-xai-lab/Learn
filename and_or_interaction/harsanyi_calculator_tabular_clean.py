import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import comb
import torchvision

# from shapley.reward_function import modified_logsoftmax
from .util import (generate_all_masks, get_reward2Iand_mat,
                            plot_curve, plot_bar, detuple)

class HarsanyiTabular():
    def __init__(self, model: nn.Module, device, reward_function: str, n_dim: int,
                 save_dir: str, seed:int) -> None:
        '''
        reward_function: ['logit' | 'output'].
        'logit' mode uses log(p / (1-p)) as reward, while 'output' mode uses output of model as reward
        '''
        self.model = model
        self.model.eval()
        self.device = device
        self.model.to(self.device)

        self.reward_function = reward_function
        # self.baseline_value = baseline_value.to(self.device)
        # self.image_size = image_size
        # self.grid_scale = grid_scale
        # self.n_grids = grid_scale ** 2
        self.n_dim = n_dim # number of grids to compute harsanyi, default is 3x4=12

        # self.batch_size = batch_size
        # self.mean = mean
        # self.std = std

        self.save_dir = save_dir # save_dir, fname, then result_name
        self.seed = seed

        self.all_masks = torch.FloatTensor(generate_all_masks(self.n_dim)).to(self.device)  # (2^n_dim, n_dim) 0-1 tensor, sorted by order
        self.reward2Iand_mat = get_reward2Iand_mat(self.n_dim).to(self.device) # (2^n_dim, 2^n_dim)


    def _get_reward(self, output, target):
        num_classes = output.shape[-1]
        if self.reward_function == 'logit':
            return output[:, target] - torch.logsumexp(output[:, np.arange(num_classes) != target], dim=1)
        elif self.reward_function == 'output':
            return output[:, target]
        else:
            raise ValueError('Invlid reward function')


    def _cal_vN(self, data, target):
        with torch.no_grad():
            output_N = self.model(data)
            output_N = detuple(output_N)
            v_N = self._get_reward(output_N, target)
        return v_N.item()

    def _cal_vEmpty(self, baseline, target):
        with torch.no_grad():
            output_Empty = self.model(baseline)
            output_Empty = detuple(output_Empty)
            v_Empty = self._get_reward(output_Empty, target)
        return v_Empty.item()

    def _save_single_file(self, file, fname, folder_name, save_name, format='npy'):
        save_root = os.path.join(self.save_dir, fname, folder_name)
        os.makedirs(save_root, exist_ok=True)
        if format == "npy":
            np.save(os.path.join(save_root, f'{save_name}.npy'), file)
        elif format == "bin":
            torch.save(file, os.path.join(save_root, f'{save_name}.bin'))
        else:
            raise NotImplementedError(f"File format [{format}] is not supported")

    def _save_IS_by_order(self, IS, fname, folder_name, save_name):
        save_root = os.path.join(self.save_dir, fname, folder_name)
        os.makedirs(save_root, exist_ok=True)
        np.save(os.path.join(save_root, f'{save_name}_all_order.npy'), IS.detach().cpu().numpy()) # save all orders together

        # for plotting
        IS_mean_per_order = []
        IS_abs_mean_per_order = []
        IS_sum_per_order = []

        start_idx, end_idx = 0, 0
        for order in range(0, self.n_dim + 1):  # from 0 to n_dim
            print(f"--------- order = {order} ----------")
            # update start_idx, end_idx
            total_num_this_order = comb(self.n_dim, order, exact=True) # exact=True to return an integer, otherwise a float is returned
            start_idx = end_idx
            end_idx = end_idx + total_num_this_order

            # slice the I(S) for specific order
            IS_this_order = IS[start_idx:end_idx].detach().cpu().numpy()
            print("IS_this_order shape:", IS_this_order.shape)
            np.save(os.path.join(save_root, f'{save_name}_order{order}.npy'), IS_this_order)

            IS_mean_per_order.append(np.mean(IS_this_order))
            IS_abs_mean_per_order.append(np.mean(np.abs(IS_this_order)))
            IS_sum_per_order.append(np.sum(IS_this_order))

        # plot I(S) vs. order, exclude order 0
        x = np.arange(1, len(IS_mean_per_order))
        # E_{|S|=m} I(S)
        plot_bar(x, IS_mean_per_order[1:], xlabel="order", ylabel="mean I(S)",
                   title=f"mean I(S) per order\n{fname}",
                   save_path=save_root, save_name=f"plot_{save_name}_mean_per_order")
        # E_{|S|=m} |I(S)|
        plot_bar(x, IS_abs_mean_per_order[1:], xlabel="order", ylabel="mean |I(S)|",
                   title=f"mean |I(S)| per order\n{fname}",
                   save_path=save_root, save_name=f"plot_{save_name}_abs_mean_per_order")
        # sum_{|S|=m} I(S)
        plot_bar(x, IS_sum_per_order[1:], xlabel="order", ylabel="sum I(S)",
                   title=f"sum I(S) per order\n{fname}",
                   save_path=save_root, save_name=f"plot_{save_name}_sum_per_order")
        # disentanglement  |E_{|S|=m} I(S)| / E_{|S|=m} |I(S)|
        disentanglement = np.abs(np.array(IS_mean_per_order)) / np.array(IS_abs_mean_per_order)
        plot_curve(x, disentanglement[1:], xlabel="order", ylabel="|E I(S)| / E |I(S)|",
                 title=f"disentanglement\n{fname}",
                 save_path=save_root, save_name=f"plot_{save_name}_disentanglement")


    def _cal_all_rewards(self, data, target, baseline):
        """
        Input:
            data: (1,n_dim)
            baseline: (1,n_dim)
        Return:
            rewards: (2^n_dim,) tensor
        """
        assert data.shape[0] == 1
        n_samples = len(self.all_masks)
        with torch.no_grad():
            data_expand = data.expand_as(self.all_masks).clone()
            baseline_expand = baseline.expand_as(self.all_masks).clone()
            masked_players = self.all_masks * data_expand + (1 - self.all_masks) * baseline_expand
            assert masked_players.shape == (n_samples, self.n_dim)
            outputs = self.model(masked_players)
            outputs = detuple(outputs)
            rewards = self._get_reward(outputs, target)
        assert rewards.shape == (n_samples,)
        return rewards

    def _check_efficiency(self, v_N, IS):
        print("v(N) = ", v_N)
        print("sum I(S) = ", IS.sum().item())


    def compute_harsanyi(self, data, target, baseline, fname):
        """
        Input:
            data: (1,n_dim) tensor
            target: int
            baseline: (1,n_dim) tensor
            fname: string, name of the sample
        Return:
        """
        data = data.to(self.device)
        baseline = baseline.to(self.device)
        assert type(target) == int, "Here target should be an integer, instead of a integer tensor"
        # target = target.to(self.device) # Here, target is int, no need to move to gpu

        # save v(N)
        v_N = self._cal_vN(data, target) # scalar
        self._save_single_file(np.array(v_N), fname, folder_name="v_N", save_name="v_N")

        # save v(Empty)
        v_Empty = self._cal_vEmpty(baseline, target) # scalar
        self._save_single_file(np.array(v_Empty), fname, folder_name="v_Empty", save_name="v_Empty")

        rewards = self._cal_all_rewards(data, target, baseline) # (2^n_dim,) tensor

        # save rewards for debugging
        self._save_single_file(rewards.detach().cpu().numpy(), fname, folder_name="rewards", save_name="rewards")

        # calculate Iand, Ior and save them
        Iand = torch.matmul(self.reward2Iand_mat, rewards) # 与交互

        print("saving Iand ...")
        self._save_IS_by_order(Iand, fname, folder_name="Iand", save_name="Iand")


        return v_N, v_Empty, rewards, Iand


