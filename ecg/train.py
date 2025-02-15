import argparse
import json
import keras
import numpy as np
import os
import random
import time

import network
import load
import util

MAX_EPOCHS = 100

def make_save_dir(dirname, experiment_name):
    start_time = f"{int(time.time())}-{random.randrange(1000)}"
    save_dir = os.path.join(dirname, experiment_name, start_time)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def get_filename_for_saving(save_dir):
    return os.path.join(save_dir, "{val_loss:.3f}-{val_acc:.3f}-{epoch:03d}-{loss:.3f}-{acc:.3f}.hdf5")

def train(args, params):
    print("Loading training set...")
    train = load.load_dataset(params['train'])
    print("Loading dev set...")
    dev = load.load_dataset(params['dev'])
    print("Building preprocessor...")
    preproc = load.Preproc(*train)
    print(f"Training size: {len(train[0])} examples.")
    print(f"Dev size: {len(dev[0])} examples.")

    save_dir = make_save_dir(params['save_dir'], args.experiment)
    util.save(preproc, save_dir)

    params.update({
        "input_shape": [None, 1],
        "num_categories": len(preproc.classes)
    })

    model = network.build_network(**params)

    stopping = keras.callbacks.EarlyStopping(patience=8)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        factor=0.1,
        patience=2,
        min_lr=params["learning_rate"] * 0.001
    )

    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath=get_filename_for_saving(save_dir),
        save_best_only=False
    )

    batch_size = params.get("batch_size", 32)

    if params.get("generator", False):
        train_gen = load.data_generator(batch_size, preproc, *train)
        dev_gen = load.data_generator(batch_size, preproc, *dev)
        model.fit(
            train_gen,
            steps_per_epoch=len(train[0]) // batch_size,
            epochs=MAX_EPOCHS,
            validation_data=dev_gen,
            validation_steps=len(dev[0]) // batch_size,
            callbacks=[checkpointer, reduce_lr, stopping]
        )
    else:
        train_x, train_y = preproc.process(*train)
        dev_x, dev_y = preproc.process(*dev)
        model.fit(
            train_x, train_y,
            batch_size=batch_size,
            epochs=MAX_EPOCHS,
            validation_data=(dev_x, dev_y),
            callbacks=[checkpointer, reduce_lr, stopping]
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    parser.add_argument("--experiment", "-e", help="tag with experiment name", default="default")
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        params = json.load(f)

    train(args, params)
