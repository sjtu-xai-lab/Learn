import os
import numpy as np
import torch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

FONT = 20

__all__ = ["generate_all_masks", "get_reward2Iand_mat", "get_reward2Ior_mat", "get_Iand2reward_mat", "get_Ior2reward_mat",
           "plot_curve", "plot_bar", "plot_interaction_descending_sort_by_strength",
           "plot_interaction_strength_descending","bar_interaction_descending_sort_by_strength"]

# sort by orders
def generate_all_masks(length: int) -> list:
    from itertools import combinations
    all_S = []
    for order in range(length+1): # 0 to length
        all_S_of_order = list(combinations(np.arange(length), order))
        all_S.extend(all_S_of_order)
    masks = np.zeros((2**length, length))
    for i, S in enumerate(all_S):
        masks[i, S] = 1
    masks = [[bool(int(item)) for item in mask] for mask in masks] # list of bools
    return masks

# original
# def generate_all_masks(length: int) -> list:
#     masks = list(range(2**length))
#     masks = [np.binary_repr(mask, width=length) for mask in masks]
#     masks = [[bool(int(item)) for item in mask] for mask in masks]
#     return masks


def set_to_index(A):
    '''
    convert a boolean mask to an index
    :param A: <np.ndarray> bool (n_dim,)
    :return: an index

    [In] set_to_index(np.array([1, 0, 0, 1, 0]).astype(bool))
    [Out] 18
    '''
    assert len(A.shape) == 1
    A_ = A.astype(int)
    return np.sum([A_[-i-1] * (2 ** i) for i in range(A_.shape[0])])


def is_A_subset_B(A, B):
    '''
    Judge whether $A \subseteq B$ holds
    :param A: <numpy.ndarray> bool (n_dim, )
    :param B: <numpy.ndarray> bool (n_dim, )
    :return: Bool
    '''
    assert A.shape[0] == B.shape[0]
    return np.all(np.logical_or(np.logical_not(A), B))


def is_A_subset_Bs(A, Bs):
    '''
    Judge whether $A \subseteq B$ holds for each $B$ in 'Bs'
    :param A: <numpy.ndarray> bool (n_dim, )
    :param Bs: <numpy.ndarray> bool (n, n_dim)
    :return: Bool
    '''
    assert A.shape[0] == Bs.shape[1]
    is_subset = np.all(np.logical_or(np.logical_not(A), Bs), axis=1)
    return is_subset


def select_subset(As, B):
    '''
    Select A from As that satisfies $A \subseteq B$
    :param As: <numpy.ndarray> bool (n, n_dim)
    :param B: <numpy.ndarray> bool (n_dim, )
    :return: a subset of As
    '''
    assert As.shape[1] == B.shape[0]
    is_subset = np.all(np.logical_or(np.logical_not(As), B), axis=1)
    return As[is_subset]


def set_minus(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    '''
    calculate A/B
    :param A: <numpy.ndarray> bool (n_dim, )
    :param B: <numpy.ndarray> bool (n_dim, )
    :return: A\B

    set_minus(A=np.array([1, 1, 0, 0, 1, 1], dtype=bool), B=np.array([1, 0, 0, 0, 0, 1], dtype=bool))
    array([False,  True, False, False,  True, False])

    set_minus(A=np.array([1, 1, 0, 0, 1, 1], dtype=bool), B=np.array([1, 0, 1, 0, 0, 1], dtype=bool))
    array([False,  True, False, False,  True, False])
    '''
    assert A.shape[0] == B.shape[0] and len(A.shape) == 1 and len(B.shape) == 1
    A_ = A.copy()
    A_[B] = False
    return A_


def get_subset(A):
    '''
    Generate the subset of A
    :param A: <numpy.ndarray> bool (n_dim, )
    :return: subsets of A

    get_subset(np.array([1, 0, 0, 1, 0, 1], dtype=bool))
    array([[False, False, False, False, False, False],
           [False, False, False, False, False,  True],
           [False, False, False,  True, False, False],
           [False, False, False,  True, False,  True],
           [ True, False, False, False, False, False],
           [ True, False, False, False, False,  True],
           [ True, False, False,  True, False, False],
           [ True, False, False,  True, False,  True]])
    '''
    assert len(A.shape) == 1
    n_dim = A.shape[0]
    n_subsets = 2 ** A.sum()
    subsets = np.zeros(shape=(n_subsets, n_dim)).astype(bool)
    subsets[:, A] = np.array(generate_all_masks(A.sum()))
    return subsets


def generate_subset_masks(set_mask, all_masks):
    '''
    For a given S, generate its subsets L's, as well as the indices of L's in [all_masks]
    :param set_mask:
    :param all_masks:
    :return: the subset masks, the bool indice
    '''
    set_mask_ = set_mask.expand_as(all_masks)
    is_subset = torch.logical_or(set_mask_, torch.logical_not(all_masks))
    is_subset = torch.all(is_subset, dim=1)
    return all_masks[is_subset], is_subset


def generate_reverse_subset_masks(set_mask, all_masks):
    '''
    For a given S, with subsets L's, generate N\L, as well as the indices of L's in [all_masks]
    :param set_mask:
    :param all_masks:
    :return:
    '''
    set_mask_ = set_mask.expand_as(all_masks)
    is_rev_subset = torch.logical_or(set_mask_, all_masks)
    is_rev_subset = torch.all(is_rev_subset, dim=1)
    return all_masks[is_rev_subset], is_rev_subset


def generate_set_with_intersection_masks(set_mask, all_masks):
    '''
    For a given S, generate L's, s.t. L and S have intersection as well as the indices of L's in [all_masks]
    :param set_mask:
    :param all_masks:
    :return:
    '''
    set_mask_ = set_mask.expand_as(all_masks)
    have_intersection = torch.logical_and(set_mask_, all_masks)
    have_intersection = torch.any(have_intersection, dim=1)
    return all_masks[have_intersection], have_intersection


##################
# The following functions are for transformation between rewards and interactions
##################

def get_reward2Iand_mat(dim):
    '''
    The transformation matrix (containing 0, 1, -1's) from reward to and-interaction (Harsanyi)
    :param dim: the input dimension n
    :return: a matrix, with shape 2^n * 2^n
    '''
    all_masks = torch.BoolTensor(generate_all_masks(dim))
    n_masks, _ = all_masks.shape
    mat = []
    for i in range(n_masks):
        mask_S = all_masks[i]
        row = torch.zeros(n_masks)
        # ===============================================================================================
        # Note: I(S) = \sum_{L\subseteq S} (-1)^{s-l} v(L)
        mask_Ls, L_indices = generate_subset_masks(mask_S, all_masks)
        L_indices = (L_indices == True).nonzero(as_tuple=False)
        assert mask_Ls.shape[0] == L_indices.shape[0]
        row[L_indices] = torch.pow(-1., mask_S.sum() - mask_Ls.sum(dim=1)).unsqueeze(1)
        # ===============================================================================================
        mat.append(row.clone())
    mat = torch.stack(mat).float()
    return mat


def get_reward2Ior_mat(dim):
    '''
    The transformation matrix (containing 0, 1, -1's) from reward to or-interaction
    :param dim: the input dimension n
    :return: a matrix, with shape 2^n * 2^n
    '''
    all_masks = torch.BoolTensor(generate_all_masks(dim))
    n_masks, _ = all_masks.shape
    mat = []
    for i in range(n_masks):
        mask_S = all_masks[i]
        row = torch.zeros(n_masks)
        # ===============================================================================================
        # Note: I(S) = -\sum_{L\subseteq S} (-1)^{s+(n-l)-n} v(N\L) if S is not empty
        if mask_S.sum() == 0:
            row[i] = 1.
        else:
            mask_NLs, NL_indices = generate_reverse_subset_masks(mask_S, all_masks)
            NL_indices = (NL_indices == True).nonzero(as_tuple=False)
            assert mask_NLs.shape[0] == NL_indices.shape[0]
            row[NL_indices] = - torch.pow(-1., mask_S.sum() + mask_NLs.sum(dim=1) + dim).unsqueeze(1)
        # ================================================================================================
        mat.append(row.clone())
    mat = torch.stack(mat).float()
    return mat


def get_Iand2reward_mat(dim):
    all_masks = torch.BoolTensor(generate_all_masks(dim))
    n_masks, _ = all_masks.shape
    mat = []
    for i in range(n_masks):
        mask_S = all_masks[i]
        row = torch.zeros(n_masks)
        # ================================================================================================
        # Note: v(S) = \sum_{L\subseteq S} I(S)
        mask_Ls, L_indices = generate_subset_masks(mask_S, all_masks)
        row[L_indices] = 1.
        # ================================================================================================
        mat.append(row.clone())
    mat = torch.stack(mat).float()
    return mat


def get_Ior2reward_mat(dim):
    all_masks = torch.BoolTensor(generate_all_masks(dim))
    n_masks, _ = all_masks.shape
    mat = []
    mask_empty = torch.zeros(dim).bool()
    _, empty_indice = generate_subset_masks(mask_empty, all_masks)
    for i in range(n_masks):
        mask_S = all_masks[i]
        row = torch.zeros(n_masks)
        # ================================================================================================
        # Note: v(S) = I(\emptyset) + \sum_{L: L\union S\neq \emptyset} I(S)
        row[empty_indice] = 1.
        mask_Ls, L_indices = generate_set_with_intersection_masks(mask_S, all_masks)
        row[L_indices] = 1.
        # ================================================================================================
        mat.append(row.clone())
    mat = torch.stack(mat).float()
    return mat


def plot_curve(x, data, xlabel, ylabel, title, save_path, save_name):
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.title(title, fontsize=FONT)
    plt.plot(x, data)
    plt.xlabel(xlabel, fontsize=FONT)
    plt.ylabel(ylabel, fontsize=FONT)
    plt.tick_params(labelsize=FONT)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{save_name}.png"), dpi=200)
    plt.close()

def plot_bar(x, data, xlabel, ylabel, title, save_path, save_name):
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.title(title, fontsize=FONT)
    plt.bar(x, data)
    plt.xlabel(xlabel, fontsize=FONT)
    plt.ylabel(ylabel, fontsize=FONT)
    plt.tick_params(labelsize=FONT)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{save_name}.png"), dpi=200)
    plt.close()

def plot_interaction_strength_descending(interaction, save_path, save_name, title="interaction strength (descending)", standard=None):
    os.makedirs(save_path, exist_ok=True)
    length = len(interaction)
    strength = np.abs(interaction)
    strength = strength[np.argsort(-strength)]

    plt.figure()
    plt.plot(np.arange(length), strength)
    if standard is not None:
        for r in [1.0, 0.1, 0.05, 0.01]:
            plt.hlines(y=r*standard, xmin=0, xmax=length-1, linestyles="dashed", colors="red")
            idx = np.where(strength <= r*standard)[0][0]
            plt.scatter(idx, strength[idx], c="red")
            plt.annotate(f"{idx}", (idx, strength[idx]), zorder=5)
    plt.title(title, fontsize=FONT)
    plt.xlabel(r"# of patterns $S$", fontsize=FONT)
    plt.ylabel(r"|I(S)|", fontsize=FONT)
    plt.tick_params(labelsize=FONT)
    plt.ylim(0, None) # strength is always >= 0
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{save_name}.png"), dpi=200)#, transparent=True)
    plt.close("all")


def plot_interaction_descending_sort_by_strength(interaction, save_path, save_name,
                                                 title="interaction (sorted by descending strength)"):
    os.makedirs(save_path, exist_ok=True)
    length = len(interaction)
    strength = np.abs(interaction)
    sort_idx = np.argsort(-strength)

    plt.figure()
    plt.plot(np.arange(length), interaction[sort_idx])
    plt.title(title, fontsize=FONT)
    plt.xlabel(r"# of patterns $S$", fontsize=FONT)
    plt.ylabel(r"I(S)", fontsize=FONT)
    plt.tick_params(labelsize=FONT)
    # plt.ylim(0, None) # strength is always >= 0
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{save_name}.png"), dpi=200)#, transparent=True)
    plt.close("all")


def bar_interaction_descending_sort_by_strength(interaction, save_path, save_name,
                                                 title="interaction (sorted by descending strength)"):
    os.makedirs(save_path, exist_ok=True)
    length = len(interaction)
    strength = np.abs(interaction)
    sort_idx = np.argsort(-strength)

    plt.figure()
    plt.bar(np.arange(length), interaction[sort_idx])
    plt.title(title, fontsize=FONT)
    plt.xlabel(r"# of patterns $S$", fontsize=FONT)
    plt.ylabel(r"I(S)", fontsize=FONT)
    plt.tick_params(labelsize=FONT)
    # plt.ylim(0, None) # strength is always >= 0
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{save_name}.png"), dpi=400)#, transparent=True)
    plt.close("all")


# def bar_pq_descending_sort_by_strength(interaction, arr, save_path, save_name, ylabel,
#                                                  title="p (sorted by descending interaction strength)"):
#     os.makedirs(save_path, exist_ok=True)
#     length = len(interaction)
#     assert len(arr) == length
#     strength = np.abs(interaction)
#     sort_idx = np.argsort(-strength)
#
#     plt.figure()
#     plt.bar(np.arange(length), arr[sort_idx])
#     plt.title(title, fontsize=FONT)
#     plt.xlabel(r"# of patterns $S$", fontsize=FONT)
#     plt.ylabel(ylabel, fontsize=FONT)
#     plt.tick_params(labelsize=FONT)
#     # plt.ylim(0, None) # strength is always >= 0
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_path, f"{save_name}.png"), dpi=400)#, transparent=True)
#     plt.close("all")




if __name__ == '__main__':
    dim = 3
    all_masks = torch.BoolTensor(generate_all_masks(dim))
    print("bool mask:", all_masks)

    reward2Iand = get_reward2Iand_mat(dim)
    print("reward2Iand:", reward2Iand)
    reward2Ior = get_reward2Ior_mat(dim)
    print("reward2Ior:", reward2Ior)
    Iand2reward = get_Iand2reward_mat(dim)
    print("Iand2reward:", Iand2reward)
    Ior2reward = get_Ior2reward_mat(dim)
    print("Ior2reward:", Ior2reward)

    print(torch.matmul(reward2Iand, Iand2reward))
    print(torch.matmul(reward2Ior, Ior2reward))
