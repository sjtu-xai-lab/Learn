import os
import os.path as osp
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random
import json
import pandas as pd
import sys
sys.path.append("..")

from tools.utils import makedirs, set_seed
from config import USED_SAMPLE_NUM, SPLIT_RATIO

SEED = 0
set_seed(SEED)
DATASET = "census"


def preprocess_census(src_folder, dst_folder):
    """ Return the Adult census data in a nice package. """
    dtypes = [
        ("Age", "float32"), ("Workclass", "category"), ("fnlwgt", "float32"),
        ("Education", "category"), ("Education-Num", "float32"), ("Marital Status", "category"),
        ("Occupation", "category"), ("Relationship", "category"), ("Race", "category"),
        ("Sex", "category"), ("Capital Gain", "float32"), ("Capital Loss", "float32"),
        ("Hours per week", "float32"), ("Country", "category"), ("Target", "category")
    ]
    raw_data = pd.read_csv(
        osp.join(src_folder, "adult.data"),
        names=[d[0] for d in dtypes],
        na_values=["?"],
        skipinitialspace=True,
        dtype=dict(dtypes)
    )
    raw_data = raw_data.dropna().reset_index(drop=True)  # drop data points that contains any N/A value
    data = raw_data.drop(["Education"], axis=1)  # redundant with Education-Num
    filt_dtypes = list(filter(lambda x: not (x[0] in ["Target", "Education"]), dtypes))
    data["Target"] = data["Target"] == ">50K"
    rcode = {
        "Not-in-family": 0,
        "Unmarried": 1,
        "Other-relative": 2,
        "Own-child": 3,
        "Husband": 4,
        "Wife": 5
    }
    for k, dtype in filt_dtypes:
        if dtype == "category":
            if k == "Relationship":
                data[k] = np.array([rcode[v.strip()] for v in data[k]])
            else:
                data[k] = data[k].cat.codes

    # ==================================================================================================================
    #    processing the data
    # ==================================================================================================================
    X = data.drop(["Target", "fnlwgt"], axis=1)
    attribute_names = X.columns.tolist()
    X = X.values
    class_name = "Target"
    y = data[class_name].values.astype(int)
    # print(X.shape, y.shape)  # (30162, 12) (30162,)
    all_data = np.hstack([y[:, None], X])

    np.random.shuffle(all_data)
    X = all_data[:, 1:]
    y = all_data[:, 0].astype(np.int)

    # normalize data
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X = (X - X_mean) / X_std

    # split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=SPLIT_RATIO[DATASET], random_state=SEED)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)  # (24129, 12) (6033, 12) (24129,) (6033,)

    print()
    print("- > train shape: {}; test shape: {}".format(X_train.shape, X_test.shape))
    print("sample of X - > {}".format(X_train[:1]))
    print("sample of y - > {}".format(y_train[:10]))

    # Sample 1000 X and y from train
    idx_lst = list(range(X_train.shape[0]))
    random.shuffle(idx_lst)
    X_train_sampled = X_train[idx_lst][:USED_SAMPLE_NUM[DATASET]]
    y_train_sampled = y_train[idx_lst][:USED_SAMPLE_NUM[DATASET]]
    print("- > sampled dataset (from train) shape: {}".format(X_train_sampled.shape))

    # Sample 1000 X and y from test
    idx_lst = list(range(X_test.shape[0]))
    random.shuffle(idx_lst)
    X_test_sampled = X_test[idx_lst][:USED_SAMPLE_NUM[DATASET]]
    y_test_sampled = y_test[idx_lst][:USED_SAMPLE_NUM[DATASET]]
    print("- > sampled dataset (from test) shape: {}".format(X_test_sampled.shape))

    makedirs(dst_folder)

    np.save(osp.join(dst_folder, "X_mean_original.npy"), X_mean)
    np.save(osp.join(dst_folder, "X_std_original.npy"), X_std)

    np.save(osp.join(dst_folder, "X_train.npy"), X_train)
    np.save(osp.join(dst_folder, "y_train.npy"), y_train)
    np.save(osp.join(dst_folder, "X_test.npy"), X_test)
    np.save(osp.join(dst_folder, "y_test.npy"), y_test)

    np.save(osp.join(dst_folder, "X_train_sampled.npy"), X_train_sampled)
    np.save(osp.join(dst_folder, "y_train_sampled.npy"), y_train_sampled)
    np.save(osp.join(dst_folder, "X_test_sampled.npy"), X_test_sampled)
    np.save(osp.join(dst_folder, "y_test_sampled.npy"), y_test_sampled)

    dataset_info = {
        "attributes": attribute_names,
        "target": class_name,
        "n_attribute": X_train.shape[1],
        "n_train_sample": X_train.shape[0],
        "n_test_sample": X_test.shape[0],
        "n_sampled_train": X_train_sampled.shape[0],
        "n_sampled_test": X_test_sampled.shape[0],
        "X_mean_original": X_mean.tolist(),
        "X_std_original": X_std.tolist()
    }

    with open(osp.join(dst_folder, "info.json"), "w") as f:
        json.dump(dataset_info, f, indent=4)



if __name__ == '__main__':
    data_folder = "/data/tabular"
    src_folder = osp.join(data_folder, "Census-Dataset-Raw")
    dst_folder = osp.join(data_folder, "census")

    preprocess_census(src_folder, dst_folder)