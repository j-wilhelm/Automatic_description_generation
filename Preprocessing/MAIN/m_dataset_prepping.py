import os
import sys

cwd = os.getcwd()

sys.path.append(cwd)

# Dataset prepping comprises various steps. This file takes a configuration, comprising a folder containing raw images,
# and a folder containing descriptions. It resizes the images. Moreover, InceptionV3 and ResNet50 are used to generate
# different features. For this, finetuned models are used based on the Ralph Lauren Dataset. Moreover, the models are
# finetuned from scratch as well. From these finetuned models, new features are extracted too. Lastly, a vocabulary is
# build, as well as different word embeddings, et cetera.

from h_dataset_prepping import DataSet_Prepping
import pickle
import getopt
import json
import traceback

# configurations
arguments = getopt.getopt(sys.argv[1:], shortopts='j:')

print(arguments)

path_to_config_file = arguments[0][0][1]

with open(path_to_config_file, 'r') as config_file:
    config = json.load(config_file)

print(config)

# Get configurations from config file
try:
    GPU_part = config['GPU_part']
    webshop_name = config['webshop_name']
    raw_imgs_folder = config['raw_imgs_folder']
    raw_anns_folder = config['raw_anns_folder']
    desc_filename_length = config['desc_filename_length']
    extract_embeddings = config['embeddings']
except KeyError:
    raise KeyError("Missing required paramater in init file")

try:
    vocab_options = config['vocab_options']
except KeyError:
    vocab_options = None

try:
    extractors = config['extractors']
except KeyError:
    extractors = "all"

try:
    testing = config['testing']
except KeyError:
    testing = "extensive"

try:
    training = config['training']
except KeyError:
    training = True

try:
    train_test_split = config['train_test_split']
    train_test_split = tuple(train_test_split)
except KeyError:
    train_test_split = (0.75, 0.05, 0.2)
print("   TTEEESSSTTTT   ")
# initialize dataset
print(extractors)
if not GPU_part:
    print("starting initalization")
    new_dataset = DataSet_Prepping(raw_imgs_folder, raw_anns_folder, webshop_name, desc_filename_length,
                                   train_test_split)
    new_dataset.vocab_options = vocab_options
    print("Building vocabulary")
    new_dataset.build_vocabulary()

    # Build the different embedding matrices
    if extract_embeddings:
        new_dataset.get_embeddings()

    # Resize the images
    print("STARTING IMAGE PREPPING")
    new_dataset.classes = new_dataset.image_prepping_cpu()
    new_dataset.nr_classes = len(new_dataset.classes)

    with open(os.path.join(cwd, "variables", webshop_name, "dataset_class.p"), 'wb') as fp:
        pickle.dump(new_dataset, fp)

elif GPU_part:
    print("   Opening dataset class...   ")
    with open(os.path.join(cwd, "variables", webshop_name, 'dataset_class.p'), 'rb') as fp:
        new_dataset = pickle.load(fp)
        new_dataset.feat_extraction_dict = {}
        # \\TODO remove this and change upon_restart at h_dataset_prepping.py
        # for mode in ["fromFF", "fromscratch"]:
        #     new_dataset.feat_extraction_dict[f"incv3_{mode}"] = os.path.join(new_dataset.weight_path,
        #                                                                      "feature_extractors",
        #                                                                     f"best_model_incv3_{mode}.h5")
        # print("   Starting image prepping...")
        try:
            new_dataset.image_prepping(training=training, extractors=extractors, testing=testing)
        except Exception as e:
            with open(os.path.join(cwd, "variables", webshop_name, "dataset_class.p"), 'wb') as x:
                pickle.dump(new_dataset, x)
            print(str(e))
            traceback.print_exc(file=sys.stdout)
