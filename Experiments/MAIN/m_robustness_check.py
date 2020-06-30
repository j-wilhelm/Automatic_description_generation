# These need to be at the top to allow for running on cluster
import gc
import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)

# Other imports
import json
import getopt
from h_captionmodel import CategoryModel
import pickle
import tensorflow.keras.backend as K

"""This file is the main file for training models for the robustness experiment. Specifically, it is different from
 regular training in the sense the one of the inputs for models is not the image but instead a clothing category.
 It does no input checking. The input is a list of JSON files.
"""
arguments = getopt.getopt(sys.argv[1:], shortopts='j:')
print(arguments)

config_files = arguments[0][0][1].split("-")
config_files = [l.strip() for l in config_files]

print(" CONFIGURATIONS:   ", config_files)
# Allows to loop over several configurations so I don't need to monitor this.
for file in config_files:
    web = file[:2].lower()
    print("  WEB    ", web)
    path_to_config_file = os.path.join(cwd, "CONFIGS", f"{web}_configs", file)
    print(" PATH   ", path_to_config_file)
    with open(path_to_config_file, 'r') as f:
        config = json.load(f)
    print(config)

    captmodel = CategoryModel(config)

    captmodel.build_model(save_plot=True)

    # Train model
    nr_epochs = config["nr_epochs"]
    batch_size = config["batch_size"]
    # for testing / debugging
    # captmodel.train_imgs = captmodel.train_imgs[:35]
    # captmodel.val_imgs = captmodel.val_imgs[:24]
    # Train using bleu for early stopping
    captmodel.train_model(batch_size, nr_epochs, k=4)
    # Save train history
    with open(os.path.join(cwd, "models", "Output", config["webshop_name"], config["output_name"] + "history.p"),
              "wb") as f:
        pickle.dump(captmodel.history, f)
    del captmodel
    K.clear_session()
    gc.collect()