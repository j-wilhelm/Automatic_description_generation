# These need to be at the top to allow for running on cluster
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
import time
import tensorflow.keras.backend as K

arguments = getopt.getopt(sys.argv[1:], shortopts='j:')
print(arguments)

# Get X random pictures and generate captions for them; save caption and picture name for each
# Get different config files
full_res_dict = {}
config_files = arguments[0][0][1].split("-")
config_files = [l.strip() for l in config_files]
print(" CONFIGURATIONS:   ", config_files)
# Allows to loop over several configurations so I don't need to monitor this.

for file in config_files:

    path_to_config_file = os.path.join(cwd, "CONFIGS", f"{file[:2].lower()}_configs", file)
    with open(path_to_config_file, 'r') as f:
        config = json.load(f)
    print(config)
    print("Initializing model...")
    captmodel = CaptionModel(config)
    print("Building architecture...")
    # Build Model
    captmodel.build_model(save_plot=False)

    weights_path = os.path.join(cwd, "models", "Weights", captmodel.webshop_name,
                                config["output_name"] + "_best_weights.h5")
    batch_size = 1
    print(weights_path)
    print("Loading weights...")
    # Load weights
    captmodel.load_weights(weights_path)
    if captmodel.architecture_type == "attention":
        wpath = os.path.join(cwd, "models", "Weights", captmodel.webshop_name)
        captmodel.inference_model.load_weights(os.path.join(wpath, config["output_name"] + "inference_best_Weights.h5"))
        captmodel.initstate_model.load_weights(os.path.join(wpath, config["output_name"] + "initstate_best_Weights.h5"))

    # Generate sample of test imgs
    img_list = captmodel.test_imgs[::50]
    img_basenames = [[os.path.basename(x[:-4]), x[-4:]] for x in img_list]

    # Load filenames dict to look up the img names
    with open(os.path.join(cwd, "variables", captmodel.webshop_name, "filenames_dict.json"), 'r') as f:
        filenames_dict = json.load(f)

    filenames_dict_rev = {v: k for k, v in filenames_dict.items()}
    img_names = [filenames_dict_rev[x[0]] + x[1] for x in img_basenames]
    results_dict = captmodel.evaluate_model(BLEU=False, ROUGE=False, img_list=img_list,
                                            beam=False, batch_size=batch_size)
    results_dict["img_names"] = img_names
    full_res_dict[file] = results_dict
with open(os.path.join(cwd, "models", "Output", f"results_dict_imgs_2.json"), 'w') as f:
    json.dump(full_res_dict, f)