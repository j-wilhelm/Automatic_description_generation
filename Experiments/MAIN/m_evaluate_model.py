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
    config["force_cropped"] = True
    captmodel = CaptionModel(config)

    # For chapter 6.4
    captmodel.force_cropped = True

    print("Building architecture...")
    # Build Model
    captmodel.build_model(save_plot=False)

    weights_path = os.path.join(cwd, "models", "Weights", captmodel.webshop_name,
                                config["output_name"] + "_best_weights.h5")
    batch_size = config["batch_size"]
    print(weights_path)
    print("Loading weights...")
    # Load weights
    captmodel.load_weights(weights_path)
    if captmodel.architecture_type == "attention":
        batch_size = 32
        wpath = os.path.join(cwd, "models", "Weights", captmodel.webshop_name)
        captmodel.inference_model.load_weights(os.path.join(wpath, config["output_name"] + "inference_best_Weights.h5"))
        captmodel.initstate_model.load_weights(os.path.join(wpath, config["output_name"] + "initstate_best_Weights.h5"))
    print(f"Evaluating model on {len(captmodel.test_imgs)} validation images")
    # Evaluate using the validation set
    start = time.time()
    # results_dict = captmodel.evaluate_model(BLEU=True, ROUGE=True, img_list=captmodel.val_imgs, nr_steps=64, beam=3)
    # with open(os.path.join(cwd, "models", "Output", captmodel.webshop_name, "results",
    #                        f"test_results_dict_{file[:-4]}json"), 'w') as f:
    #     json.dump(results_dict, f)
    # end_beam = time.time()
    # print(f"Predicting val set with beam took {end_beam - start} time.")


    results_dict = captmodel.evaluate_model(BLEU=True, ROUGE=True, img_list=captmodel.test_imgs,
                                            beam=False, batch_size=batch_size)

    end_greedy = time.time()
    print(f"Predicting val set with greedy took {end_greedy - start} time.")
    # Save results_dict
    with open(os.path.join(cwd, "models", "Output", captmodel.webshop_name, "results",
                           f"results_dict{file[:-4]}_cropped.json"), 'w') as f:
        json.dump(results_dict, f)
    K.clear_session()

