import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)

from h_evaluate_model import ModelPrediction
from tensorflow.keras.layers import Input
import os
import sys
from h_buildModels import build_model, build_incv3_feat, build_resnet50_feat, build_par_inject_model, build_brownlee_model
import numpy as np
import json
import pickle
import getopt


class InvalidArgumentError(Exception):
    pass


class ConfigurationMissingError(Exception):
    pass


arguments = getopt.getopt(sys.argv[1:], shortopts='j:')

print(arguments)
path_to_config_file = arguments[0][0][1]

with open(path_to_config_file, 'r') as config_file:
    config = json.load(config_file)

print(config)

# Get configurations from config file
try:
    sentence_length = config['sentence_length']
    img_folder = config['img_folder']
    weight_folder = config['weight_folder']
    feature_extractor = config['feature_extractor']
    batch_size = config['batch_size']
    nr_nodes = config['nr_nodes']
    output_name = config['output_name']
    dropout = config['dropout']
    epoch_start = config['epoch_start']
    nr_epochs = config['nr_epochs']
    desc_filename_length = config['desc_filename_length']
    img_dim = tuple(config['img_dim'])
    optimizer = config['optimizer']
    epoch_end = config['epoch_end']
    epoch_interval = config['epoch_interval']
    architecture_type = config['architecture_type']
    image_manipulation = config['image_manipulation']
    resume = config['resume']

except KeyError as e:
    raise ConfigurationMissingError("You seem to be missing a configuration option in your file. Expected the following"
                                    "key: {}".format(str(e)))

# Always the same for RL stuff. Might be easier to keep this and make a new file for FF. If all datasets have to be
# evaluated a more general method should be found
input_path_desc = os.path.join(cwd, "variables", "RL")
with open(os.path.join(input_path_desc, "RL_embedding_matrix_200_thr10.txt"), "r", encoding="utf8") as f:
    embedding_matrix = np.loadtxt(f)
with open(os.path.join(input_path_desc, "RL_wordtoix_thr10.txt"), "r", encoding="utf8") as f:
    wordtoix = json.load(f)
with open(os.path.join(input_path_desc, "RL_ixtoword_thr10.txt"), "r", encoding="utf8") as f:
    ixtoword = json.load(f)
with open(os.path.join(input_path_desc, "list_test_imgs_RL.p"), "rb") as f:
    list_test_imgs = pickle.load(f)

model_path = os.path.join(cwd, "models")
output_path = os.path.join(model_path, "Output", "RL")
descriptions_RL_path = os.path.join(cwd, "Datasets", "RL", "Descriptions_RL")

# Paths which are dependent on the configuration
img_feature_path = os.path.join(cwd, "Datasets", "RL", img_folder)
weights_path = os.path.join(model_path, "Weights", "RL", weight_folder)

vocabulary_size = len(wordtoix) + 1
if feature_extractor == "incv3":
    cnn_input = build_incv3_feat()
elif feature_extractor == "resnet50":
    cnn_input = build_resnet50_feat()

if feature_extractor == 'incv3':
    list_test_imgs_correct = []
    for i in list_test_imgs:
        list_test_imgs_correct.append(i[:-1] + ".p")
else:
    list_test_imgs_correct = list_test_imgs

preds = []
bleus = []
rouges = []

if resume:
    # Load dictionary and extract preds, bleus and rouges to resume predictions
    with open(os.path.join(output_path, output_name + "_evaluations_intermediates.json"), 'r') as f:
        results_dict = json.load(f)

    bleus = results_dict["bleu_scores"]
    preds = results_dict["predictions"]
    rouges = results_dict["rouge_scores"]
if architecture_type == "merge":
    model = build_model(cnn_input, cnn_input, sentence_length, vocabulary_size=vocabulary_size,
                        embedding_matrix=embedding_matrix, concat_add='concatenate')

elif architecture_type == "parinject":
    model = build_par_inject_model(cnn_input, sentence_length, vocabulary_size, embedding_matrix,
                                   embedding_dimensions=200, optimizer=optimizer,
                                   image_manipulation=image_manipulation, nodes_per_layer=nr_nodes,
                                   dropout=dropout)

elif architecture_type == "brownlee":
    model = build_brownlee_model(cnn_input, sentence_length, vocabulary_size, embedding_matrix,
                                   embedding_dimensions=200, optimizer=optimizer,
                                   image_manipulation=image_manipulation, nodes_per_layer=nr_nodes,
                                   dropout=dropout)

try:
    for i in range(epoch_start, epoch_end, epoch_interval):
        if i == 0:
            continue
        print("Starting new iteration for weights with epoch {}".format(i))
        model_name = output_name + "_epoch" + str(i) + ".h5"
        model.load_weights(os.path.join(weights_path, model_name))

        RL_preds = ModelPrediction(list_test_imgs_correct, img_feature_path, descriptions_RL_path, desc_filename_length,
                                   model, wordtoix, ixtoword)
        RL_preds.make_predictions()
        bleus.append((i, RL_preds.bleu_scores))
        rouges.append((i, RL_preds.rouge_scores))
        preds.append((i, RL_preds.predictions))
        intermediate_results_dict = {'bleu_scores': bleus, 'rouge_scores': rouges, 'predictions': preds}
        with open(os.path.join(output_path, output_name + "_evaluations_intermediates.json"), 'w') as f:
            json.dump(intermediate_results_dict, f)

    results_dict = {'bleu_scores': bleus, 'rouge_scores': rouges, 'predictions': preds}
    with open(os.path.join(output_path, output_name + "_evaluations.json"), 'w') as f:
        json.dump(results_dict, f)

except Exception as e:
    print("exception occured")
    print(str(e))
    results_dict = {'bleu_scores': bleus, 'rouge_scores': rouges, 'predictions': preds}
    with open(os.path.join(output_path, output_name + "_evaluations_intermediates.json"), 'w') as f:
        json.dump(results_dict, f)
