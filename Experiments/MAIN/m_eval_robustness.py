# These need to be at the top to allow for running on cluster
import os
import random
import sys

cwd = os.getcwd()
sys.path.append(cwd)

# Other imports
import json
import getopt
from h_captionmodel import CategoryModel
import pickle
import time
import tensorflow.keras.backend as K

arguments = getopt.getopt(sys.argv[1:], shortopts='j:')
print(arguments)

# Get different config files

config_files = arguments[0][0][1].split("-")
config_files = [l.strip() for l in config_files]

print(" CONFIGURATIONS:   ", config_files)
# Allows to loop over several configurations so I don't need to monitor this.
for file in config_files:
    path_to_config_file = os.path.join(cwd, "CONFIGS", f"{file[:2].lower()}_configs", file)
    #
    with open(path_to_config_file, 'r') as f:
        config = json.load(f)
    print(config)
    print("Initializing model...")
    config["force_cropped"] = True if config["attribute_included"] else False
    captmodel = CategoryModel(config)
    print("Building architecture...")
    # Build Model
    captmodel.build_model(save_plot=False)

    weights_path = os.path.join(cwd, "models", "Weights", captmodel.webshop_name,
                                config["output_name"] + "_best_weights.h5")
    batch_size = config["batch_size"] * 4
    print(weights_path)
    print("Loading weights...")
    # Load weights
    captmodel.load_weights(weights_path)
    start = time.time()
    results_dict = captmodel.evaluate_model(BLEU=True, ROUGE=True, img_list=captmodel.test_imgs,
                                            beam=False, batch_size=batch_size)

    end_greedy = time.time()
    print(f"Predicting val set with greedy took {end_greedy - start} time.")
    # Save results_dict
    with open(os.path.join(cwd, "models", "Output", captmodel.webshop_name, "results",
                           f"results_dict{file[:-4]}json"), 'w') as f:
        json.dump(results_dict, f)
    K.clear_session()