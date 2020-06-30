# These need to be at the top to allow for running on cluster
import gc
import os
import random
import sys

cwd = os.getcwd()
sys.path.append(cwd)

# Other imports
import json
import getopt
from h_captionmodel import CaptionModel
import pickle
import tensorflow as tf
import tensorflow.keras.backend as K
arguments = getopt.getopt(sys.argv[1:], shortopts='j:')
print(arguments)

config_files = arguments[0][0][1].split("-")
config_files = [l.strip() for l in config_files]

print(" CONFIGURATIONS:   ", config_files)
# Allows to loop over several configurations so I don't need to monitor this.
for file in config_files:
    print(file)
    with open(os.path.join(cwd, "CONFIGS", "co_configs", file), 'r') as f:
        config = json.load(f)

    # Get train, val and test images:
    with open(os.path.join(cwd, "variables", "combined", "train_imgs.p"), 'rb') as f:
        train_imgs = pickle.load(f)
    with open(os.path.join(cwd, "variables", "combined", "val_imgs.p"), 'rb') as f:
        val_imgs = pickle.load(f)
    with open(os.path.join(cwd, "variables", "combined", "test_imgs.p"), 'rb') as f:
        test_imgs = pickle.load(f)

    config["train_imgs"] = train_imgs
    config["val_imgs"] = val_imgs
    config["test_imgs"] = test_imgs

    captmodel = CaptionModel(config)
    # captmodel.nr_webshops = config["nr_webshops"]
    captmodel.build_model()
    best_weight_path = os.path.join(cwd, "models", "Weights", config["webshop_name"],
                                        config["output_name"] + "_best_weights.h5")
    if os.path.exists(best_weight_path):
        captmodel.load_weights(best_weight_path)
    nr_epochs = config["nr_epochs"]
    batch_size = 32

    try:
        es = config["early_stopping"]
    except KeyError:
        es = True
    print(es)
    if not es:
        # use BLEU for early stopping
        use_validation = True
        validation_steps = False
        k = 3
    else:
        use_validation = False
        validation_steps = False
        k = None
    # captmodel.train_imgs = random.sample(captmodel.train_imgs, 96)
    # captmodel.val_imgs = random.sample(captmodel.train_imgs, 32)
    captmodel.train_model(nr_epochs, batch_size, start_epoch=0, use_validation=use_validation,
                              validation_steps=validation_steps,
                                k=k, early_stopping=es)
    with open(os.path.join(cwd, "models", "Output", config["webshop_name"], config["output_name"] + "history.p"),
              "wb") as f:
        pickle.dump(captmodel.history, f)

    del captmodel
    K.clear_session()
    gc.collect()
