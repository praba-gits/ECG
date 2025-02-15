import json
import numpy as np
import os
import random
import scipy.io as sio
import tqdm

STEP = 256


def load_ecg_mat(ecg_file):
    return sio.loadmat(ecg_file)['val'].squeeze()


def load_all(data_path):
    label_file = os.path.join(data_path, "REFERENCE-v3.csv")  # Corrected path

    if not os.path.exists(label_file):
        raise FileNotFoundError(f"REFERENCE-v3.csv not found at {label_file}")

    with open(label_file, 'r') as fid:
        records = [l.strip().split(",") for l in fid]

    dataset = []
    for record, label in tqdm.tqdm(records):
        ecg_file = os.path.join(data_path, record + ".mat")
        ecg_file = os.path.abspath(ecg_file)

        if not os.path.exists(ecg_file):
            print(f"Warning: ECG file {ecg_file} not found. Skipping...")
            continue

        ecg = load_ecg_mat(ecg_file)
        num_labels = len(ecg) // STEP  # Integer division
        dataset.append((ecg_file, [label] * num_labels))
    return dataset


def split(dataset, dev_frac):
    dev_cut = int(dev_frac * len(dataset))
    random.shuffle(dataset)
    dev = dataset[:dev_cut]
    train = dataset[dev_cut:]
    return train, dev


def make_json(save_path, dataset):
    with open(save_path, 'w') as fid:
        for d in dataset:
            datum = {'ecg': d[0], 'labels': d[1]}
            json.dump(datum, fid)
            fid.write('\n')


if __name__ == "__main__":
    random.seed(2018)

    dev_frac = 0.1
    data_path = "/content/ECG/examples/cinc17/data/"

    dataset = load_all(data_path)
    train, dev = split(dataset, dev_frac)

    make_json("train.json", train)
    make_json("dev.json", dev)

    print("✅ Train and dev JSON files created successfully!")
