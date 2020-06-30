# These need to be at the top to allow for running on cluster
import os
import random
import sys
import time

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
for config_name in config_files:
    with open(os.path.join(cwd, "CONFIGS", "co_configs", config_name), 'r') as f:
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
    batch_size = 32
    print(f"Evaluating model on {len(captmodel.test_imgs)} validation images")
    start = time.time()
    results_dict = captmodel.evaluate_model(BLEU=True, ROUGE=True, img_list=captmodel.test_imgs,
                                                beam=False, batch_size=batch_size)

    end_greedy = time.time()
    print(f"Predicting val set with greedy took {end_greedy - start} time.")
    # Save results_dict
    with open(os.path.join(cwd, "models", "Output", captmodel.webshop_name, "results",
                           f"results_dict_{config_name}"), 'w') as f:
        json.dump(results_dict, f)
    K.clear_session()