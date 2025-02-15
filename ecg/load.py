import json
import keras
import numpy as np
import os
import random
import scipy.io as sio
import tqdm

STEP = 256


def data_generator(batch_size, preproc, x, y):
    num_examples = len(x)
    examples = list(zip(x, y))  # Convert zip object to list
    examples = sorted(examples, key=lambda x: x[0].shape[0])
    end = num_examples - batch_size + 1
    batches = [examples[i:i + batch_size] for i in range(0, end, batch_size)]
    random.shuffle(batches)

    while True:
        for batch in batches:
            x, y = zip(*batch)
            yield preproc.process(x, y)


class Preproc:
    def __init__(self, ecg, labels):
        self.mean, self.std = compute_mean_std(ecg)
        self.classes = sorted(set(l for label in labels for l in label))
        self.int_to_class = dict(zip(range(len(self.classes)), self.classes))
        self.class_to_int = {c: i for i, c in self.int_to_class.items()}

    def process(self, x, y):
        return self.process_x(x), self.process_y(y)

    def process_x(self, x):
        x = pad(x)
        x = (x - self.mean) / self.std
        x = x[:, :, None]
        return x

    def process_y(self, y):
        y = pad([[self.class_to_int.get(c, 0) for c in s] for s in y], val=3, dtype=np.int32)
        y = keras.utils.to_categorical(y, num_classes=len(self.classes))
        return y


def pad(x, val=0, dtype=np.float32):
    max_len = max(len(i) for i in x)
    padded = np.full((len(x), max_len), val, dtype=dtype)
    for e, i in enumerate(x):
        padded[e, :len(i)] = i
    return padded


def compute_mean_std(x):
    x = np.hstack(x)
    return np.mean(x).astype(np.float32), np.std(x).astype(np.float32)


def load_dataset(data_json):
    if not os.path.exists(data_json):
        raise FileNotFoundError(f"Error: {data_json} not found!")

    with open(data_json, 'r') as fid:
        data = [json.loads(l) for l in fid]

    labels = []
    ecgs = []

    for d in tqdm.tqdm(data, desc="Loading Data"):
        labels.append(d['labels'])
        ecgs.append(load_ecg(d['ecg']))

    return ecgs, labels


def load_ecg(record):
    if not os.path.exists(record):
        raise FileNotFoundError(f"ECG file not found: {record}")

    ext = os.path.splitext(record)[1].lower()

    if ext == ".npy":
        ecg = np.load(record)
    elif ext == ".mat":
        mat_data = sio.loadmat(record)
        if 'val' in mat_data:
            ecg = mat_data['val'].squeeze()
        else:
            raise ValueError(f"Unexpected format in MAT file: {record}")
    else:  # Assumes binary 16-bit integers
        with open(record, 'rb') as fid:  # Use 'rb' for binary mode
            ecg = np.fromfile(fid, dtype=np.int16)

    trunc_samp = STEP * (len(ecg) // STEP)
    return ecg[:trunc_samp]


if __name__ == "__main__":
    data_json = "/content/ECG/examples/cinc17/train.json"  # Ensure correct path
    try:
        train = load_dataset(data_json)
        preproc = Preproc(*train)
        gen = data_generator(32, preproc, *train)

        for x, y in gen:
            print(f"✅ Batch X Shape: {x.shape}, Batch Y Shape: {y.shape}")
            print(f"🔍 First 5 Labels: {y[:5]}")
            break
    except Exception as e:
        print(f"❌ Error: {e}")
