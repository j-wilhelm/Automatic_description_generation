# These need to be at the top to allow for running on cluster
import os
import sys
import trace

cwd = os.getcwd()
sys.path.append(cwd)

# Other imports
import json
import random
import getopt
from h_captionmodel import CaptionModel
from h_utils import masked_categorical_crossentropy
import pickle
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

"""This file is the main file for training caption models. It does no input checking. The input is a list of JSON
"""

# NR available GPUs:
nr_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
print(f" USING {nr_gpus}  GPUs")

# \\ TODO add argument so foldername can also be included
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

    # Initialize CaptModel
    captmodel = CaptionModel(config)
    captmodel.loss_function = masked_categorical_crossentropy
    captmodel.optimizer = Adam(0.001)
    # Build Model
    save_plot = config["save_plot"] if "save_plot" in config.keys() else True
    captmodel.build_model(save_plot=save_plot, nr_gpus=nr_gpus)

    # Train model
    nr_epochs = config["nr_epochs"]
    batch_size = config["batch_size"] / 2
    try:
        es = config["early_stopping"]
    except KeyError:
        es = True
    print(es)
    if not es:
        # use BLEU for early stopping
        use_validation = True
        validation_steps = False
        k = int(nr_epochs / 5)
    else:
        use_validation = False
        validation_steps = False
        k = None

    # Check whether previous weights are available. If so; load them:
    weights_path = os.path.join(cwd, "models", "Weights", config["webshop_name"])
    best_weight_path = os.path.join(cwd, "models", "Weights", config["webshop_name"],
                                    config["output_name"] + "_best_weights.h5")
    c_weight_path = os.path.join(weights_path, config["output_name"] + "_checkp_weights.h5")
    if os.path.exists(best_weight_path) & os.path.exists(c_weight_path):
        lm_weights = os.path.getmtime(best_weight_path)
        if lm_weights > os.path.getmtime(c_weight_path):
            weight_name = "best"
        else:
            weight_name = "checkp"
        weight_name = "best"

        l_weight_path = os.path.join(weights_path, config["output_name"] + f"_{weight_name}_weights.h5")
    # captmodel.weight_path = weights_path
    # captmodel.current_epoch = 1
        captmodel.load_weights(l_weight_path)
        captmodel.inference_model.load_weights(os.path.join(weights_path,
                                                            captmodel.output_name + f"inference_{weight_name}_Weights.h5"))
        captmodel.initstate_model.load_weights(os.path.join(weights_path,
                                                            captmodel.output_name + f"initstate_{weight_name}_Weights.h5"))
        # Load validation results
        # with open(os.path.join(cwd, "models", "Output", config["webshop_name"], config["output_name"] + "_Last_VAL_RESULTS.json"),
        #          "r") as f:
        #     try:
        #         captmodel.val_results = json.load(f)
        #     except EOFError:
        #         captmodel.val_results = []


    # captmodel.train_imgs = random.sample(captmodel.train_imgs, 96)
    # captmodel.val_imgs = random.sample(captmodel.train_imgs, 32)
    # Run in normal mode
    captmodel.train_model(nr_epochs, batch_size=16, start_epoch=0, use_validation=True, validation_steps=False, k=5,
                          early_stopping=False)


    # save history and val results
    with open(os.path.join(cwd, "models", "Output", config["webshop_name"], config["output_name"] + "history.p"),
              "wb") as f:
        pickle.dump(captmodel.history, f)
    with open(os.path.join(cwd, "models", "Output", config["webshop_name"], config["output_name"] + "val_results.p"),
              "wb") as f:
        pickle.dump(captmodel.val_results, f)

    K.clear_session()

#
# modulelist = ["training", "nest", "type_spec", "tensor_shape", "dtypes", "backend", "ops", "threading",
#               "iostream", "socket", "tracking", "python_message", "tf_stack", "type_checkers", "function",
#               "contextlib", "pywrap_tensorflow_internal", "garbage", "decoder", "structure", "dataset_ops",
#               "c_api_util", "tensor_spec", "lock_util", "traceable_stack", "control_flow_util", "layer_utils",
#               "objct_identity", "layer_utils", "data_adapter", "context", "gen_dataset_ops", "base", "inspect",
#               "tf_decorator", "__init__", "func_graph", "functools", "containers", "compat", "encoder",
#               "descriptor", "api", "abc", "inspect_utils", "op_def_library", "gen_random_ops", "execute",
#               "random_seed", "random_ops", "message", "_collections_abc", "op_def_library", "conversion",
#               "config_lib", "constant_op", "math_ops", "gen_array_ops", "object_identity", "auto_control_deps",
#               "tape", "message_listener", "tensor_util", "auto_control_deps", "graph_only_ops", "os",
#               "data_structures", "distribution_strategy_context", "training_v2_utils", "composite_tensor_utils",
#               "training_utils", "deprecation", "tensor_conversion_registry", "weakref", "variable_scope",
#               "tf_context_lib", "api_implementation", "converter", "ag_logging", "options", "resource_variable_ops",
#               "variables", "optimization_options", "array_ops", "dispatch", "gen_math_ops", "six",
#               "tf_contextlib", "memory", "tf_logging", "training_v2", "callbacks", "gen_nn_ops", "backprop",
#               "numerictypes", "custom_gradient", "network", "ag_ctx", "distribute_lib", "from_numeric", "core",
#               "utils", "function_base", "generic_utils", "_bootstrap_external", "enum", "function_utils",
#               "_bootstrap", "tf_inspect", "losses", "tf2", "numeric", "_methods", "fromnumeric", "device",
#               "gen_resource_variable", "device_spec", "device", "gen_resource_variable_ops", "server_lib",
#               "distributed_training_utils", "monitoring", "composite_tensor", "embedding_ops", "embeddings",
#               "h_customGenerator", "CustomGenerator", "sequence", "random", "training_generator", "data_utils", "heap",
#               "sharedctypes", "tempfile", "warnings", "registry", "math_grad", "types", "_asarray", "common_shapes",
#               "re", "nn_ops", "tensor_array_ops", "h_utils", "arrayprint", "tf_should_use"]

# Run in debug mode
# tracer = trace.Trace(ignoremods=modulelist)
# tracer.run("captmodel.train_model(nr_epochs, batch_size=8, start_epoch=0, use_validation=True, validation_steps=False, k=1,early_stopping=False)")