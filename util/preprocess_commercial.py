import os
import os.path as osp
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random
import json
import sys
sys.path.append("..")

from tools.utils import makedirs, set_seed
from config import USED_SAMPLE_NUM, SPLIT_RATIO

SEED = 0
set_seed(SEED)
DATASET = "commercial"
SELECTED_DIMENSIONS = {
    1: "shot length",
    2: "motion",
    4: "frame difference",
    6: "short time energy",
    8: "zero crossing rate",
    10: "spectral centroid",
    12: "spectral roll off",
    14: "spectral flux",
    16: "fundamental frequency",
    4124: "edge change ratio"
}

def preprocess_commercial(src_paths, dst_folder):
    makedirs(dst_folder)

    raw_data = []
    for src_path in src_paths:
        with open(src_path, "r") as f:
            raw_data.extend(f.readlines())
    print("# of data:", len(raw_data))
    raw_data = [item.split() for item in raw_data]

    for i in tqdm(range(len(raw_data)), desc="processing"):
        label = int(raw_data[i][0])
        if label == -1: label = 0
        features = [item.split(":") for item in raw_data[i][1:]]
        features = [(int(k), float(v)) for k, v in features]
        features = dict(features)
        features = [features[k] for k in SELECTED_DIMENSIONS.keys()]
        raw_data[i] = [label] + features

    raw_data = np.array(raw_data)

    print("Ratio of positive samples:", np.mean(raw_data[:, 0] == 1))
    print("Ratio of negative samples:", np.mean(raw_data[:, 0] == 0))

    # To balance positive/negative samples
    print("-> Balancing positive/negative samples")
    pos_data = raw_data[raw_data[:, 0] == 1]
    neg_data = raw_data[raw_data[:, 0] == 0]
    sample_num = pos_data.shape[0] if pos_data.shape[0] < neg_data.shape[0] else neg_data.shape[0]
    pos_data = pos_data[:sample_num]
    neg_data = neg_data[:sample_num]

    raw_data = np.concatenate((pos_data, neg_data), axis=0)
    print("Ratio of positive samples:", np.mean(raw_data[:, 0] == 1))
    print("Ratio of negative samples:", np.mean(raw_data[:, 0] == 0))

    np.random.shuffle(raw_data)
    X = raw_data[:, 1:]
    y = raw_data[:, 0].astype(np.int)
    # normalize data
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X = (X - X_mean) / X_std

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=SPLIT_RATIO[DATASET], random_state=SEED)
    print("- > train shape: {}; test shape: {}".format(X_train.shape, X_test.shape))

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
        "attributes": list(SELECTED_DIMENSIONS.values()),
        "target": "is_commercial",
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

    return



if __name__ == '__main__':
    data_folder = "/data/tabular"
    data_files = [
        "TV_News_Channel_Commercial_Detection_Dataset/CNN.txt",
        "TV_News_Channel_Commercial_Detection_Dataset/BBC.txt",
    ]

    preprocess_commercial(
        src_paths=[osp.join(data_folder, data_file) for data_file in data_files],
        dst_folder=osp.join(data_folder, "commercial")
    )