import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import comb
import torchvision
from typing import List, Union, Tuple

from .util import (generate_all_masks, get_reward2Iand_mat, plot_curve, plot_bar)

class HarsanyiImg():
    def __init__(self, model: nn.Module, device, reward_function: str,
                 image_size: int, grid_scale: int, n_dim: int, batch_size: int, mean, std,
                 save_dir: str, seed:int, background="baseline") -> None:
        '''
        reward_function: ['logit' | 'output'].
        'logit' mode uses log(p / (1-p)) as reward, while 'output' mode uses output of model as reward
        background: ['baseline' | 'ori']
        'baseline' mode means we mask all patches apart from the patches to calculate Harsanyi
        'ori' mode means we keep all patches apart from the patches to calculate Harsanyi to their original values
        '''
        assert reward_function in ["logit", "output"]
        assert background in ["baseline", "ori"]
        self.model = model
        self.model.eval()
        self.device = device
        self.model.to(self.device)

        self.reward_function = reward_function
        self.background = background
        self.image_size = image_size
        self.grid_scale = grid_scale
        self.n_grids = grid_scale ** 2
        self.n_dim = n_dim # number of grids to compute harsanyi, default is 3x4=12

        self.batch_size = batch_size
        self.mean = mean
        self.std = std

        self.save_dir = save_dir # save_dir, fname, then result_name
        self.seed = seed

        self.region_masks = torch.FloatTensor(generate_all_masks(self.n_dim)).to(self.device)  # (2^n_dim, n_dim) 0-1 tensor, sorted by order
        self.reward2Iand_mat = get_reward2Iand_mat(self.n_dim).to(self.device) # (2^n_dim, 2^n_dim)


    def _get_reward(self, output, target):
        num_classes = output.shape[-1]
        if self.reward_function == 'logit':
            return output[:, target] - torch.logsumexp(output[:, np.arange(num_classes) != target], dim=1)
        elif self.reward_function == 'output':
            return output[:, target]
        else:
            raise ValueError('Invlid reward function')


    def _cal_vN(self, image, target, region_idx):
        # here v_N refers to the chosen 3x4 region, not the whole image
        mask = self._get_N_mask(region_idx)
        with torch.no_grad():
            masked_image = mask * image + (1-mask) * self.baseline_value
            output_N = self.model(masked_image)
            v_N = self._get_reward(output_N, target)
        return v_N.item()

    def _cal_vEmpty(self, image, target, region_idx):
        mask = self._get_Empty_mask(region_idx)
        with torch.no_grad():
            masked_image = mask * image + (1 - mask) * self.baseline_value
            output_Empty = self.model(masked_image)
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

    def _get_N_mask(self, region_idx):
        """ get the mask for v(N)
        background = "baseline": mask is 1 for grids in region_idx, 0 for grids not in region_idx
        background = "ori": mask is all-one
        """
        if self.background == "baseline":
            mask = torch.zeros(1, self.n_grids, device=self.device)
            # set 1 for the chosen grids, while keep zero for other grids
            mask[:, region_idx] = 1.
        elif self.background == "ori":
            mask = torch.ones(1, self.n_grids, device=self.device)
        else:
            raise NotImplementedError(f"background mode [{self.background}] not implemented")
        mask = mask.reshape(-1, 1, self.grid_scale, self.grid_scale)
        mask = F.interpolate(mask.clone(), size=[self.image_size, self.image_size], mode="nearest").float()
        return mask

    def _get_Empty_mask(self, region_idx):
        """ get the mask for v(empty)
        background = "baseline": mask is all-zero
        background = "ori": mask is 0 for grids in region_idx, 1 for grids not in region_idx
        """
        if self.background == "baseline":
            mask = torch.zeros(1, self.n_grids, device=self.device)
        elif self.background == "ori":
            mask = torch.ones(1, self.n_grids, device=self.device)
            mask[:, region_idx] = 0
        else:
            raise NotImplementedError(f"background mode [{self.background}] not implemented")
        mask = mask.reshape(-1, 1, self.grid_scale, self.grid_scale)
        mask = F.interpolate(mask.clone(), size=[self.image_size, self.image_size], mode="nearest").float()
        return mask

    def _get_all_masks(self, region_idx):
        """
        Input:
            region_idx: list of length (n_dim)
        Return:
            all_masks: (2^n_dim, 1, image_size, image_size) float tensor, only containing 0 and 1
        """
        if self.background == "baseline":
            all_masks = torch.zeros(2**self.n_dim, self.n_grids, device=self.device)
        elif self.background == "ori":
            all_masks = torch.ones(2 ** self.n_dim, self.n_grids, device=self.device)
        else:
            raise NotImplementedError(f"background mode [{self.background}] not implemented")

        # set the mask for the chosen grids
        all_masks[:, region_idx] = self.region_masks # (2^n_dim, n_dim) 0-1 tensor,
        all_masks = all_masks.reshape(-1, 1, self.grid_scale, self.grid_scale)
        all_masks = F.interpolate(all_masks.clone(), size=[self.image_size, self.image_size], mode="nearest").float()
        assert all_masks.shape[0] == 2**self.n_dim
        return all_masks


    def _cal_all_rewards(self, image, target, all_masks):
        """
        Input:
            all_masks: (2^n_dim, 1, image_size, image_size) float tensor, only containing 0 and 1
        Return:
            rewards: (2^n_dim,) tensor
        """
        assert image.shape[0] == 1
        n_samples = len(all_masks)
        rewards = []
        batch_num = int(n_samples * 1.0 / self.batch_size)
        if n_samples % self.batch_size > 0:
            batch_num += 1
        for batch in range(batch_num):
            st = batch * self.batch_size
            ed = (batch + 1) * self.batch_size
            if ed > n_samples:
                ed = n_samples
            cnt = ed - st
            expand_players = image.expand(cnt, -1, self.image_size, self.image_size).clone()
            expand_baselines = self.baseline_value.expand(cnt, -1, self.image_size, self.image_size).clone()

            # replace these masked players with baseline values
            masked_players = all_masks[st:ed] * expand_players + (1 - all_masks[st:ed]) * expand_baselines
            with torch.no_grad():
                output = self.model(masked_players)
                reward = self._get_reward(output, target)
                rewards.append(reward)
        rewards = torch.cat(rewards, dim=0) # should be (n_samples,)
        assert rewards.shape == (n_samples,)
        return rewards

    def _check_efficiency(self, v_N, v_Empty, IS):
        print("v(N) = ", v_N)
        print("sum I(S) = ", IS.sum().item())
        print("v(empty) = ", v_Empty)
        print("IS[0] = ", IS[0])

    def _visualize_image(self, image, region_idx, fname):
        mask_N = self._get_N_mask(region_idx)
        mask_Empty = self._get_Empty_mask(region_idx)
        masked_image_N = mask_N * image + (1 - mask_N) * self.baseline_value
        masked_image_Empty = mask_Empty * image + (1 - mask_Empty) * self.baseline_value
        # image is unnormalized, no need to denormalize here
        # denormalized_image = denormalize(image, mean=self.mean, std=self.std)
        # denormalized_masked_image = denormalize(masked_image, mean=self.mean, std=self.std)
        save_root = os.path.join(self.save_dir, fname, "visual")
        os.makedirs(save_root, exist_ok=True)
        torchvision.utils.save_image(image, os.path.join(save_root, "image.png"), nrow=1)
        torchvision.utils.save_image(masked_image_N, os.path.join(save_root, "image_N.png"), nrow=1)
        torchvision.utils.save_image(masked_image_Empty, os.path.join(save_root, "image_Empty.png"), nrow=1)

    def compute_harsanyi(self, image, target, baseline_value, region_idx, fname, visual=False):
        assert len(region_idx) == self.n_dim, "The length of region_idx is not equal to the n_dim argument!"
        image = image.to(self.device)
        assert type(target) == int, "Here target should be an integer, instead of a integer tensor"
        # target = target.to(self.device) # Here, target is int, no need to move to gpu

        if image.shape[1] == 3 and visual: # input image instead of feature map, then visualize
            self._visualize_image(image.clone().detach(), region_idx, fname)

        self.baseline_value = baseline_value.to(self.device)

        # save region_idx
        self._save_single_file(np.array(region_idx), fname, folder_name="region_idx", save_name="region_idx")

        # save v(N)
        v_N = self._cal_vN(image, target, region_idx) # scalar
        self._save_single_file(np.array(v_N), fname, folder_name="v_N", save_name="v_N")

        # save v(Empty)
        v_Empty = self._cal_vEmpty(image, target, region_idx) # scalar
        self._save_single_file(np.array(v_Empty), fname, folder_name="v_Empty", save_name="v_Empty")

        # calculate all rewards of the subsets
        # The rewards are sorted in the following way:
        # v(empty),
        # v({1}), v({2}), ..., v({n}),
        # v({1,2}), v({1,3}), v({1,4}), ..., v({n-1,n}),
        # ...
        # v({1,2,...,n})
        all_masks = self._get_all_masks(region_idx)
        rewards = self._cal_all_rewards(image, target, all_masks) # (2^n_dim,) tensor

        # save rewards for debugging
        self._save_single_file(rewards.detach().cpu().numpy(), fname, folder_name="rewards", save_name="rewards")

        # calculate Iand, Ior and save them
        Iand = torch.matmul(self.reward2Iand_mat, rewards)
        print("saving Iand ...")
        self._save_IS_by_order(Iand, fname, folder_name="Iand", save_name="Iand")

        return v_N, v_Empty, rewards, Iand

