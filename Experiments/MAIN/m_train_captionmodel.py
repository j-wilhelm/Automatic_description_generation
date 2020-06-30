# These need to be at the top to allow for running on cluster
import gc
import os
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

"""This file is the main file for training caption models. It does no input checking. The input is a list of JSON
"""

# NR available GPUs:
nr_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
print(f" USING {nr_gpus}  GPUs")

arguments = getopt.getopt(sys.argv[1:], shortopts='j:')
print(arguments)

config_files = arguments[0][0][1].split("-")
config_files = [l.strip() for l in config_files]

print(" CONFIGURATIONS:   ", config_files)
# Allows to loop over several configurations so I don't need to monitor this.
for file in config_files:
    try:
        web = file[:2].lower()
        print("  WEB    ", web)
        path_to_config_file = os.path.join(cwd, "CONFIGS", f"{web}_configs", file)
        print(" PATH   ", path_to_config_file)
        with open(path_to_config_file, 'r') as f:
            config = json.load(f)
        print(config)

        captmodel = CaptionModel(config)

        # Build Model
        save_plot = config["save_plot"] if "save_plot" in config.keys() else True
        captmodel.build_model(save_plot=save_plot, nr_gpus=nr_gpus)

        # Check whether previous weights are available. If so; load them:
        # best_weight_path = os.path.join(cwd, "models", "Weights", config["webshop_name"],
        #                                 config["output_name"] + "_best_weights.h5")
        # if os.path.exists(best_weight_path):
        #     captmodel.load_weights(best_weight_path)

        # Train model
        nr_epochs = config["nr_epochs"]
        batch_size = config["batch_size"]
        if (batch_size == 8) & (config["webshop_name"] not in ["MADEWELL", "RALPH_LAUREN"]):
            batch_size = 16

        if (batch_size == 16) & (config["webshop_name"] == "MADEWELL"):
            batch_size = 8
        try:
            es = config["early_stopping"]
        except KeyError:
            es = True
        print(es)
        if not es:
            # use BLEU for early stopping
            use_validation = True
            validation_steps = False
            k = 4
        else:
            use_validation = False
            validation_steps = False
            k = None


        # Use early stopping based on bleu scores or validation (generally bleu)

        captmodel.train_model(nr_epochs, batch_size, start_epoch=0, use_validation=use_validation,
                              validation_steps=validation_steps,
                                k=k, early_stopping=es)

        # Save change in batch size if relevant:
        # if nr_gpus > 1:
        #     config["batch_size"] = nr_gpus * batch_size
        #     with open(path_to_config_file, 'w') as f:
        #         json.dump(config, f)

        # Save train history
        with open(os.path.join(cwd, "models", "Output", config["webshop_name"], config["output_name"] + "history.p"),
                  "wb") as f:
            pickle.dump(captmodel.history, f)
        del captmodel
        K.clear_session()
        gc.collect()
    except Exception as e:
        print(e)
        del captmodel
        gc.collect()
        K.clear_session
        continue
