# These need to be at the top to allow for running on cluster
import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)

# Other imports
import numpy as np
import json
import math
from h_buildModels import build_par_inject_model, build_incv3_feat, build_resnet50_feat, build_brownlee_model, \
    build_attention_model, build_basic_model, ExternalAttentionRNNWrapper, build_webshopincluded_model, \
    build_category_brownlee_model, build_category_merge_model, build_category_parinject_model, \
    build_img_cat_brownlee_model, build_img_cat_merge_model, build_img_cat_parinject_model
from h_customGenerator import CustomGenerator, BLEU_validation, CategoricGenerator, AttributeGenerator
import pickle
from tensorflow.keras.utils import plot_model
import tensorflow as tf
from h_utils import get_desc, compute_corpusbleu, compute_ROUGE, masked_categorical_crossentropy
import random
from nltk.translate.bleu_score import corpus_bleu
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
from tqdm import tqdm
import gc  # garbage collection due to keras memory leak
from tensorflow.keras.models import load_model, save_model
import time
import trace


# Type testing function
def ensure_type(value, type):
    if not isinstance(value, type):
        raise TypeError(f'Value {value} is of type {value.type} but should be of type {type}')


class CaptionModel:

    def __init__(self, config):
        """
        Paramaters
        -----------
        config : dict
             dictionary object with all necessary information.
             Key webshop_name : str
                Foldername of the webshop
             Key ... : int
                ddd
        """
        cwd = os.getcwd()

        # Set attributes
        self.webshop_name = config["webshop_name"]
        self.weight_path = os.path.join(cwd, "models", "Weights", self.webshop_name)
        self.output_path = os.path.join(cwd, "models", "Output", self.webshop_name)
        self.var_path = os.path.join(cwd, "variables", self.webshop_name)

        self.img_path = os.path.join(cwd, "Datasets", self.webshop_name, config["img_folder"])
        desc_folder = os.path.join(cwd, "Datasets", self.webshop_name, "Descriptions")
        if not os.path.isdir(desc_folder):
            self.desc_path = os.path.join(cwd, "Datasets", self.webshop_name, config["desc_folder"])
        else:
            self.desc_path = desc_folder

        self.embedding_weights_path = os.path.join(self.var_path, config["embedding_name"])
        self.trainable_embedding = False  # Can be overruled if provided in the config
        self.sentence_length = config["sentence_length"]
        self.nr_nodes = config["nr_nodes"]
        self.output_name = config["output_name"]
        self.dropout = config["dropout"]
        self.desc_filename_length = config["desc_filename_length"]
        self.threshold = config["threshold"]
        self.optimizer = config["optimizer"]
        self.force_cropped = False
        self.architecture_type = config["architecture_type"]
        try:
            self.feat_extractor = config["feature_extractor"]
        except KeyError:
            self.feat_extractor = "incv3"
        self.img_dim = config["img_dim"]
        self.loss_function = "categorical_crossentropy"
        self.current_epoch = 0
        self.history = []
        self.val_results = []
        self.evaluations = []
        self.predictions = []
        self.check_print = None
        self.model = None
        if self.architecture_type[:17] == "multiple_webshops":
            self.webshop_embedding = config["webshop_embedding"]
            current_index = 0
            self.webshop_dict = {}
            all_descs = os.listdir(self.desc_path)
            for i in all_descs:
                if i[:2] not in self.webshop_dict.keys():
                    self.webshop_dict[i[:2]] = current_index
                    current_index += 1
        else:
            self.webshop_embedding = None
            self.webshop_dict = {}

        # Set optional variables; also makes it possible to override information above if needed
        necessary_key_list = ["sentence_length", "img_folder", "webshop_name", "embedding_name",
                              "nr_nodes", "output_name", "dropout", "epoch_start", "desc_filename_length",
                              "optimizer", "architecture_type"]

        # *args
        remaining_keys = [key for key in config.keys() if key not in necessary_key_list]
        for key in remaining_keys:
            setattr(self, key, config[key])

        try:
            self.feat_dims = tuple(self.feat_dims)
        except AttributeError:
            self.feat_dims = None

        # Load external variables
        if self.embedding_weights_path.endswith(".txt"):
            with open(self.embedding_weights_path, "r", encoding="utf8") as f:
                self.embedding_matrix = np.loadtxt(f)
        elif self.embedding_weights_path.endswith(".p"):
            with open(self.embedding_weights_path, "rb") as f:
                self.embedding_matrix = pickle.load(f)
        else:
            raise ValueError(f"This file extension is currently not supported. Please choose '.p' or '.txt'")

        with open(os.path.join(self.var_path, f"wordtoix_thr{self.threshold}.json"), "r", encoding="utf8") as f:
            self.wordtoix = json.load(f)

        with open(os.path.join(self.var_path, f"ixtoword_thr{self.threshold}.json"), "r", encoding="utf8") as f:
            self.ixtoword = json.load(f)

        # Define vocab size
        self.vocab_size = len(self.wordtoix) + 1

        # Infer partitions from folder structure if explicit paths are not passed in config file
        if "train_imgs" not in config.keys():
            self.train_imgs = []
            self.test_imgs = []
            self.val_imgs = []
            if ("bb" in self.img_path) | (self.force_cropped == True):
                main_img_path = os.path.join(cwd, "Datasets", self.webshop_name, "cropped_images")
                print(main_img_path)
            else:
                main_img_path = os.path.join(cwd, "Datasets", self.webshop_name, "resized_imgs")
            for cat in os.listdir(os.path.join(main_img_path, "TRAIN")):
                for img in os.listdir(os.path.join(main_img_path, "TRAIN", cat)):
                    self.train_imgs.append(img[:-4] + ".p")

            for cat in os.listdir(os.path.join(main_img_path, "VAL")):
                for img in os.listdir(os.path.join(main_img_path, "VAL", cat)):
                    self.val_imgs.append(img[:-4] + ".p")

            for cat in os.listdir(os.path.join(main_img_path, "TEST")):
                for img in os.listdir(os.path.join(main_img_path, "TEST", cat)):
                    self.test_imgs.append(img[:-4] + ".p")

            self.train_imgs = [os.path.join(self.img_path, img) for img in self.train_imgs]
            self.val_imgs = [os.path.join(self.img_path, img) for img in self.val_imgs]
            self.test_imgs = [os.path.join(self.img_path, img) for img in self.test_imgs]
        else:
            self.train_imgs = config["train_imgs"]
            self.val_imgs = config["val_imgs"]
            self.test_imgs = config["test_imgs"]

        if self.architecture_type == "attention":
            self.generator_type = "sentence"
            if self.feat_extractor == "incv3":
                self.feat_dims = (8, 8, 2048)
            elif self.feat_extractor == "resnet50":
                self.feat_dims = (7, 7, 2048)
        else:
            self.generator_type = "word"

        # Assert optional parameters are sufficiently filled
        # if features are of shape (x, x, y) --> architecture should be attention

        # Assert optimizer
        if self.feat_extractor.lower() not in ["resnet50", "incv3"]:
            if self.feat_extractor.lower().replace("_", "") == "inceptionv3":
                self.feat_extractor = "incv3"
            elif self.feat_extractor.lower().replace("_", "") == "resnet50":
                self.feat_extractor = "resnet50"
            else:
                raise ValueError(f"The provided feature extractor cannot be evaluated. {self.feat_extractor} "
                                 f"was supplied while either 'resnet50' or 'incv3' is expected.")



    def build_model(self, save_plot=True, nr_gpus=1):
        """Function which builds the model.
        Builds model based on architecture type and compiles model as well. If plot_model is true a png plot of the
        model is saved to the output location.
        Attributes
        -----------
        save_plot : bool
            whether or not to save a png plot of the file to default output location (default is True)

        """

        if self.feat_extractor == "incv3":
            cnn_input = build_incv3_feat()
        elif self.feat_extractor == "resnet50":
            cnn_input = build_resnet50_feat()

        if self.architecture_type == "brownlee":
            print(self.sentence_length, self.embedding_matrix.shape, self.vocab_size)
            self.model = build_brownlee_model(cnn_input, self.sentence_length, self.vocab_size, self.embedding_matrix,
                                              embedding_dimensions=self.embedding_matrix.shape[1],
                                              loss_function=self.loss_function, nodes_per_layer=self.nr_nodes,
                                              dropout=self.dropout, optimizer=self.optimizer,
                                              trainable_embedding=self.trainable_embedding)

        elif self.architecture_type == "parinject":
            self.model = build_par_inject_model(cnn_input, self.sentence_length, self.vocab_size, self.embedding_matrix,
                                                embedding_dimensions=self.embedding_matrix.shape[1],
                                                loss_function=self.loss_function, nodes_per_layer=self.nr_nodes,
                                                dropout=self.dropout, optimizer=self.optimizer,
                                                trainable_embedding=self.trainable_embedding)

        elif self.architecture_type == "attention":
            self.model, self.inference_model, \
            self.initstate_model = build_attention_model(self.vocab_size, nr_nodes=self.nr_nodes,
                                                         embedding_dimensions=self.embedding_matrix.shape[1],
                                                         embedding_matrix=self.embedding_matrix,
                                                         max_capt_length=self.sentence_length, feat_dims=self.feat_dims,
                                                         loss_function=self.loss_function, optimizer=self.optimizer)

        elif self.architecture_type == "merge":
            self.model = build_basic_model(cnn_input, cnn_input, self.sentence_length, self.vocab_size,
                                           self.embedding_matrix, self.embedding_matrix.shape[1], self.loss_function,
                                           self.optimizer, nr_nodes=self.nr_nodes, concat_add='add',
                                           trainable_embedding=self.trainable_embedding, dropout=self.dropout)

        elif self.architecture_type == "multiple_webshops_merge":
            self.model = build_webshopincluded_model(cnn_input, cnn_input, self.sentence_length, self.vocab_size,
                                                     self.embedding_matrix, self.embedding_matrix.shape[1],
                                                     self.nr_webshops, self.loss_function,
                                                     self.optimizer, nr_nodes=self.nr_nodes, concat_add=self.concat_add,
                                                     concat_moment=self.concat_moment,
                                                     trainable_embedding=self.trainable_embedding,
                                                     webshop_embedding=self.webshop_embedding)
        elif self.architecture_type == "multiple_webshops_parinject":
            self.model = build_img_cat_parinject_model(self.nr_webshops, self.sentence_length, self.vocab_size,
                                                       self.embedding_matrix, self.embedding_matrix.shape[1],
                                                       concat_add="concatenate",
                                                       trainable_embedding=self.trainable_embedding,
                                                       dropout=False, cat_embedding=True)
        elif self.architecture_type == "multiple_webshops_brownlee":
            self.model = build_img_cat_parinject_model(self.nr_webshops, self.sentence_length, self.vocab_size,
                                                       self.embedding_matrix, self.embedding_matrix.shape[1],
                                                       trainable_embedding=self.trainable_embedding,
                                                       dropout=False, cat_embedding=True)
        else:
            raise ValueError(f"No valid architecture type supplied. Architecture type {self.architecture_type} is "
                             f"currently not supported. Supported architecture types are 'brownlee', 'merge', "
                             f"'parinject', multiple_webshops or 'attention'.")
        print("MODEL SUMMARY    ")
        print(self.model.summary())
        if save_plot:
            plot_model(self.model, os.path.join(self.output_path, self.output_name + "_model.png"))

    def load_weights(self, external_weights_path):
        """Loads weights from an external file.
        Should only be used as a starting position. If the model resumes training, weights are automatically loaded
        based on the latest weight instance.
        Attributes
        ----------
        external_weights_path : path or str
            path to external weights file (should be either .h5 or .hdf5 extension)"""
        self.model.load_weights(external_weights_path)

    def train_model(self, nr_epochs, batch_size=16, start_epoch=0, use_validation=False, validation_steps=100,
                    k=3, save_intermediate_weights=False, early_stopping=False):
        """

        :param early_stopping : bool
            whether to apply early stopping using validation loss (default is False)
        :param save_intermediate_weights : bool
            whether to save weights for each k epochs as a seperate file or not (default is False)
        :param nr_epochs : int
            Number of epochs to train for
        :param batch_size : int
            Number of images to use in one batch (default is 16)
        :param start_epoch : int
            Start epoch; specify if continuing training (default is 0)
        :param use_validation: bool
            Whether to apply validation or not (default is False)
        :param validation_steps: int
            How many validation steps per validation iteration (default is 100)
        :param k: int
            After how many epochs to validate
        """

        # Get generators
        # Select k images from validation set; ensure same ones are selected

        print(use_validation)
        if use_validation & validation_steps:
            random.seed(0)
            self.val_imgs = random.sample(self.val_imgs, k=validation_steps)
            print(len(self.val_imgs))
        if self.architecture_type in ["attention", "brownlee"]:
            print("CHANGING QUEUE SIZE ")
            queue_size = 2
        else:
            queue_size = 10
        self.train_generator = CustomGenerator(self.train_imgs, batch_size, self.desc_path, self.vocab_size,
                                               self.wordtoix,
                                               self.sentence_length, n_channels=3, img_dim=self.img_dim,
                                               image_encoder=self.feat_extractor,
                                               txtfilelength=self.desc_filename_length,
                                               generator_type=self.generator_type, model_type=self.architecture_type,
                                               feat_dims=self.feat_dims, webshop_embedded=self.webshop_embedding,
                                               webshop_dict=self.webshop_dict)
        steps_per_epoch = math.floor(len(self.train_imgs) / batch_size)

        if start_epoch > 0:
            self.current_epoch = start_epoch
            self.load_weights(os.path.join(self.weight_path, self.output_name + f"_weights_{start_epoch + 1}"))

        if early_stopping:
            self.val_generator = CustomGenerator(self.val_imgs, batch_size, self.desc_path, self.vocab_size,
                                                 self.wordtoix,
                                                 self.sentence_length, n_channels=3, img_dim=self.img_dim,
                                                 image_encoder=self.feat_extractor,
                                                 txtfilelength=self.desc_filename_length,
                                                 generator_type=self.generator_type, model_type=self.architecture_type,
                                                 feat_dims=self.feat_dims)
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2, restore_best_weights=True)
            # mc = ModelCheckpoint(os.path.join(self.weight_path, self.output_name + "weights_{epoch:02d}"),
            #                      monitor='val_loss', save_best_only=True)
            history = self.model.fit_generator(self.train_generator, epochs=nr_epochs, verbose=1, callbacks=[es],
                                               steps_per_epoch=steps_per_epoch, validation_data=self.val_generator)
            self.history = history.history

            self.model.save_weights(os.path.join(self.weight_path, self.output_name + "_best_weights.h5"))
            self.current_epoch = nr_epochs

        else:
            for epoch in range(self.current_epoch, self.current_epoch + nr_epochs):
                print(" CURRENT EPOCH   ", self.current_epoch)
                # Train for one epoch
                print("TRAINING MODEL >>>   ")
                t = time.time()

                history = self.model.fit_generator(self.train_generator, epochs=1, verbose=1,
                                                   steps_per_epoch=steps_per_epoch, max_queue_size=queue_size)

                dt = time.time() - t
                print(f"FINISHED EPOCH took {dt} seconds")

                # Save weights
                if save_intermediate_weights:
                    self.model.save_weights(os.path.join(self.weight_path,
                                                         self.output_name + f"_weights_{epoch + 1}.h5"))
                self.current_epoch += 1

                # Apply validation
                print(" Evaluating ON EPOCH END   ")
                es = self.__on_epoch_end(use_validation, k, batch_size=batch_size, last_epoch=nr_epochs - 1)

                try:
                    self.history.append((epoch, k, history.history, self.val_results[-1]))
                except IndexError:
                    print("Can't add val results to history..", self.val_results)
                    self.history.append((epoch, k, history.history, []))
                    del history

                if es:
                    break
                # If no early stopping is applied save the weights after every epoch

        print(f"  FINISHED TRAINING after {self.current_epoch} epochs")

        # Delete remaining files
        os.remove(f"testm_{self.output_name}.h5")
        os.remove(f"tmpoptimizerweights_{self.output_name}.p")
        if self.architecture_type == "attention":
            os.remove(f"testi_{self.output_name}.h5")
            os.remove(f"testt_{self.output_name}.h5")

    def reset_training(self):
        """Resets all training progress.
        """
        self.model.save_weights(f"testm_{self.output_name}.h5")
        # Save optimizer state
        symbolic_weights_m = getattr(self.model.optimizer, 'weights')
        weight_values = K.batch_get_value(symbolic_weights_m)
        with open(f"tmpoptimizerweights_{self.output_name}.p", 'wb') as f:
            pickle.dump(weight_values, f)
        del self.model
        K.clear_session()
        gc.collect()
        self.build_model()
        self.model.load_weights(f"testm_{self.output_name}.h5")
        # Load optimizer weights
        with open(f"tmpoptimizerweights_{self.output_name}.p", "rb") as f:
            optimizer_weights = pickle.load(f)
        # self.model.optimizer.set_weights(optimizer_weights)

    def reset_training_att_val(self, batch_size):
        """Clears the keras session and restarts it partly to avoid OO< due to memory leakage."""

        self.model.save_weights(f"testm_{self.output_name}.h5")
        self.inference_model.save_weights(f"testi_{self.output_name}.h5")
        self.initstate_model.save_weights(f"testt_{self.output_name}.h5")

        # Save optimizer state
        symbolic_weights_m = getattr(self.model.optimizer, 'weights')
        weight_values = K.batch_get_value(symbolic_weights_m)
        with open(f"tmpoptimizerweights_{self.output_name}.p", 'wb') as f:
            pickle.dump(weight_values, f)

        K.clear_session()
        gc.collect()
        del self.model
        del self.initstate_model
        del self.inference_model
        del self.train_generator
        K.clear_session()
        gc.collect()
        self.build_model()
        self.model.load_weights(f"testm_{self.output_name}.h5")
        self.inference_model.load_weights(f"testi_{self.output_name}.h5")
        self.initstate_model.load_weights(f"testt_{self.output_name}.h5")
        self.train_generator = CustomGenerator(self.train_imgs, batch_size, self.desc_path, self.vocab_size,
                                               self.wordtoix,
                                               self.sentence_length, n_channels=3, img_dim=self.img_dim,
                                               image_encoder=self.feat_extractor,
                                               txtfilelength=self.desc_filename_length,
                                               generator_type=self.generator_type, model_type=self.architecture_type,
                                               feat_dims=self.feat_dims)
        # Load optimizer weights
        with open(f"tmpoptimizerweights_{self.output_name}.p", "rb") as f:
            optimizer_weights = pickle.load(f)
        self.model.optimizer.set_weights(optimizer_weights)

    def __on_epoch_end(self, use_validation, k, batch_size, last_epoch):
        """Includes several functionalities to apply on epoch end. k is taken as an interval. It both validates and
        saves weights if used.

        :param use_validation : bool
            Whether to apply validation using BLEU scores
        :param k : int
            Interval for when on epoch end has to be applied
        :return:
        """
        # Run validation data set without beam search to get BLEU scores and save weights if necessary
        if use_validation:
            if ((self.current_epoch - 1) % k == 0) | (self.current_epoch == 1) | (self.current_epoch == last_epoch):
                if self.architecture_type != "attention":
                    print("VALIDATING>>>")
                    predictions, references, _, _ = self.make_predictions(self.val_imgs, batch_size=batch_size)

                else:
                    predictions, references, _, _ = self.make_predictions_att(self.val_imgs, batch_size=batch_size)
                for p, r in zip(predictions[-5:], references[-5:]):
                    print("PREDS  ", p, "REF  ", r, "   ")
                print("  COMPUTING BLEU  ")
                score_1, score_2, score_3, score_4 = compute_corpusbleu(references, predictions)
                print("  VALIDATION  :  ", [score_1, score_2, score_3, score_4], "  ")
                self.val_results.append([score_1, score_2, score_3, score_4])

                if len(self.val_results) > 1:
                    # Check for early stopping
                    if self.val_results[-1][0] <= 0.9 * (self.val_results[-2][0]):
                        print(f" STOPPING TRAINING - Performance no longer improved during epoch. Validation "
                              f"performance in epoch {self.current_epoch} was {self.val_results[-1][0]} "
                              f"versus {self.val_results[-2][0]} in epoch {self.current_epoch - k}")
                        early_stopping = True
                        return early_stopping

                    # Also stop early if no improvement over the last 1 or 2 epochs
                    if (self.val_results[-1][0] <= (self.val_results[-2][0])) & (len(self.val_results) > 2):
                        if self.val_results[-1][0] <= (self.val_results[-3][0]):
                            print(f" STOPPING TRAINING - Performance no longer improved during epoch. Validation "
                                  f"performance in epoch {self.current_epoch} was {self.val_results[-1][0]} "
                                  f"versus {self.val_results[-3][0]} in epoch {self.current_epoch - k * 2}")
                            early_stopping = True
                            return early_stopping

                self.model.save_weights(os.path.join(self.weight_path, self.output_name + "_best_weights.h5"))
                # Also save inference and init state models
                if self.architecture_type == "attention":

                    self.inference_model.save_weights(os.path.join(self.weight_path,
                                                                   self.output_name + "inference_best_Weights.h5"))
                    self.initstate_model.save_weights(os.path.join(self.weight_path,
                                                                   self.output_name + "initstate_best_Weights.h5"))
                    last_results = {}
                    for i in range(len(references)):
                        last_results[i] = {}
                        last_results[i]["Pred"] = predictions[i]
                        last_results[i]["Ref"] = references[i]
                    with open(os.path.join(self.output_path, self.output_name + "_Last_VAL_RESULTS.json"), 'w') as f:
                        json.dump(last_results, f)
                    # restart some training parts to avoid memory leakage and OOM
                    self.reset_training_att_val(batch_size)

            elif self.architecture_type == "attention":
                self.model.save_weights(os.path.join(self.weight_path, self.output_name + "_checkp_weights.h5"))
                self.inference_model.save_weights(os.path.join(self.weight_path,
                                                               self.output_name + "inference_checkp_Weights.h5"))
                self.initstate_model.save_weights(os.path.join(self.weight_path,
                                                               self.output_name + "initstate_checkp_Weights.h5"))

                self.reset_training_att_val(batch_size)
            else:
                self.reset_training()
        else:
            self.model.save_weights(os.path.join(self.weight_path, self.output_name + "_best_weights.h5"))
            self.inference_model.save_weights(os.path.join(self.weight_path,
                                                           self.output_name + "inference_best_Weights.h5"))
            self.initstate_model.save_weights(os.path.join(self.weight_path,
                                                           self.output_name + "initstate_best_Weights.h5"))
            self.reset_training_att_val(batch_size)

    def evaluate_model(self, BLEU=True, ROUGE=True, nr_steps=None, img_list=None, beam=False, batch_size=1):
        """Method to evaluate a model.
        Makes predictions based on the test images, or another image set if specified. Returns a dictionary containing
        results.
        :param beam:
        :param BLEU : bool
            Whether or not to compute BLEU scores (1-4) (default is True)
        :param ROUGE : bool
            Whether or not to compute ROUGE scores (default is True)
        :param nr_steps : int
            Number of test steps; randomly chosen (default is None)
        :param img_list : list(str) or list(path)
            Files to use as test imgs. If not specified the test set is used. (default is None)

        :return: results : dict
            Dictionary containing, BLEU, ROUGE, predictions and references
        """

        if img_list:
            test_imgs = img_list
        else:
            test_imgs = self.test_imgs

        random.seed(0)
        if nr_steps:
            test_imgs = random.sample(test_imgs, k=nr_steps)

        print("MAKING PREDICTIONS...")
        if self.architecture_type == "attention":
            predictions, references, candidates, img_ids = self.make_predictions_att(test_imgs, batch_size)
        else:
            predictions, references, candidates, img_ids = self.make_predictions(test_imgs, beam, batch_size)

        if BLEU:
            print("Computing BLEU...")
            b1, b2, b3, b4 = compute_corpusbleu(references, predictions)
            BLEUS = {"BLEU_1": b1, "BLEU_2": b2, "BLEU_3": b3, "BLEU_4": b4}
        else:
            BLEUS = []

        predictions = [" ".join(pred) for pred in predictions]
        references = [[" ".join(ref[0])] for ref in references]

        if ROUGE:
            print("Computing ROUGE...")
            ROUGES_avg = compute_ROUGE(predictions, references, aggregator="Avg")
            ROUGES_best = compute_ROUGE(predictions, references, aggregator="Best")
            ROUGES = {"Avg": ROUGES_avg, "Best": ROUGES_best}
        else:
            ROUGES = []
        print(BLEUS, ROUGES)
        results = {"BLEU_SCORES": BLEUS, "ROUGE_SCORES": ROUGES, "IMG_IDS": img_ids, "PREDICTIONS": predictions,
                   "REFERENCES": references,
                   "CANDIDATES": candidates}

        return results

    def make_predictions_att(self, img_paths, batch_size=1):
        """Method which makes predictions for a set of images.
        :param img_paths : list
            List of image paths
        :param batch_size : int
            Size of batches to use during predictions. Higher makes the predictions faster but may result in resource
            allocation problems

        :returns predictions, references, candidates
            predictions and references are matched index wise. Candidates is an empty list for now; this changes if
            beam search is used.
        """
        # Initialization
        endseq_ix = self.wordtoix["endseq"]
        predictions = []
        references = []
        candidates = []
        print("MAKING PREDICTIONS >>> ")

        # Split the data into batches
        batch_paths = [img_paths[x: x + batch_size] for x in range(0, len(img_paths), batch_size)]

        # Loop over batches
        for batch in tqdm(batch_paths):
            # get relevant train data
            descs = []
            features = []
            input_captions = []

            # Load data
            for img_path in batch:
                with open(img_path, "rb") as f:
                    img_feat = pickle.load(f)
                img_feat = np.array([img_feat])
                img_feat = img_feat.reshape((1, 64, 2048))
                features.append(img_feat)

                desc_name = os.path.basename(img_path)[:self.desc_filename_length] + ".txt"
                desc = get_desc(os.path.join(self.desc_path, desc_name), prediction=True)
                descs.append([desc])

                input_captions.append([self.wordtoix["startseq"]])

            # Squeeze features and captions
            features = np.array(features).squeeze(axis=1)

            # Get initial states for all inputs:
            h_states, c_states = self.initstate_model.predict(features)

            # initialize attentions
            # Loop for sentence length
            for w in range(self.sentence_length - 1):
                # Make arrays of captions, this is a bit messy due to indexing and not knowing the final sentence length
                caption_arrays = np.array([np.array(x) for x in input_captions])
                # Make predictions for next word
                outputs, h_states, c_states, _ = self.inference_model.predict([features, caption_arrays,
                                                                               h_states, c_states])

                indices_to_remove = []
                # Extract words
                for ix, pred in enumerate(outputs):
                    next_word = np.argmax(pred[0])
                    input_captions[ix].append(next_word)

                    # If final end token, pop this caption and add to list of final captions and descs
                    if next_word == endseq_ix:
                        seq = [self.ixtoword[str(word)] for word in input_captions[ix][1:w + 1]]
                        indices_to_remove.append(ix)
                        predictions.append(seq)
                        references.append(descs[ix])

                # Evaluate features to be deleted
                features = np.delete(features, indices_to_remove, axis=0)
                h_states = np.delete(h_states, indices_to_remove, axis=0)
                c_states = np.delete(c_states, indices_to_remove, axis=0)

                # Delete descs and captions as well, in reverse order so previous indices are unaffected
                for d_ix in sorted(indices_to_remove, reverse=True):
                    del descs[d_ix]
                    del input_captions[d_ix]

                # If all sentences are finished, break out of the loop
                if len(input_captions) == 0:
                    break

            # Add sentences which ran on for complete duration
            if len(input_captions) != 0:
                for ix, capt in enumerate(input_captions):
                    seq = [self.ixtoword[str(word)] for word in capt[1:w + 1]]
                    predictions.append(seq)
                    references.append(descs[ix])

        return predictions, references, candidates, []

    def make_predictions(self, img_paths, beam=False, batch_size=1):
        """Makes predictions for all images provided in img paths. For this, it

        :param beam:
        :param img_paths : list(str or path)
            list of paths to images to make predictions for."""
        predictions = []
        references = []
        candidates = []
        j = 0

        if beam:
            for path in tqdm(img_paths):
                seq, desc, candidates_pred = self.__make_beam_prediction(path, beam)
                candidates.append(candidates_pred)
                predictions.append(seq[1:-1])
                references.append([desc])
        else:
            # Divide into chunks of size batch size
            batches = [img_paths[x: x + batch_size] for x in range(0, len(img_paths), batch_size)]
            for batch in tqdm(batches):
                descs, seqs = self.__make_greedy_prediction(batch)

                predictions.extend(seqs)
                references.extend(descs)
        return predictions, references, candidates, []

    def __make_greedy_prediction(self, paths_batch):
        """
        using batches greatly reduces time to make predictions for all images. Up to a max of batch_size times. In
        reality this is a bit slower, as not all items are finished at the same time.
        :param paths_batch: paths to images to make greedy predictions for.
        :return: predictions and references
        """
        endseq_ix = self.wordtoix["endseq"]
        descs = []
        features = []

        # Load descriptions (references) and image features
        for path in paths_batch:
            with open(path, "rb") as f:
                feature = pickle.load(f)

            desc_name = os.path.basename(path)[:self.desc_filename_length] + ".txt"
            path_to_desc = os.path.join(self.desc_path, desc_name)
            desc = get_desc(path_to_desc, prediction=True)
            if not feature.shape == (1, 2048):
                feature = np.reshape(feature, (1, 2048))
            descs.append(desc)
            features.append(feature)
        try:
            features = np.array(features).squeeze(axis=1)
        except ValueError:
            pass

        # Initialize empty lists to story final results in
        final_yhats = []
        final_ypreds = []

        # Make start sequences for predictions
        startseq = [self.wordtoix["startseq"]]
        startseq_pred = sequence.pad_sequences([startseq], maxlen=self.sentence_length, padding='post')
        seq_preds = [startseq_pred for i in range(len(paths_batch))]
        seq_preds = np.array(seq_preds).squeeze(axis=1)

        # Add webshop info if necessary
        if self.architecture_type[:17] == "multiple_webshops":
            webshops_preds = [self.webshop_dict[os.path.basename(x)[:2]] for x in paths_batch]
            if self.webshop_embedding == False:
                webshops_preds = to_categorical(webshops_preds, num_classes=len(self.webshop_dict))
            else:
                webshops_preds = [[w] for w in webshops_preds]
                webshops_preds = np.array(webshops_preds)

        for i in range(self.sentence_length - 1):
            indices_to_remove = []
            # Make prediction; add webshop info if necessary
            if self.architecture_type[:17] == "multiple_webshops":
                ypreds = self.model.predict([webshops_preds, seq_preds, features], batch_size=len(paths_batch))
            else:
                ypreds = self.model.predict([features, seq_preds], batch_size=len(paths_batch))

            # Get max probabilities; use this as new index
            ypreds = np.argmax(ypreds, axis=-1)

            for ix, prediction in enumerate(ypreds):

                # Add prediction to current sequence
                seq_preds[ix][i + 1] = prediction

                if prediction == endseq_ix:
                    # if endseq is reached; add prediction and ytrue to finished stuff and pop from sequences to predict
                    # transform to sentence
                    # Transform all class labels up and including endseq to words
                    seq = [self.ixtoword[str(w)] for w in seq_preds[ix][1:i + 1]]
                    final_ypreds.append(seq)
                    final_yhats.append([descs[ix]])
                    indices_to_remove.append(ix)

            # Remove all finished descriptions; in reverse order so previous indices are not disrupted
            seq_preds = np.delete(seq_preds, indices_to_remove, axis=0)
            features = np.delete(features, indices_to_remove, axis=0)
            if self.architecture_type[:17] == "multiple_webshops":
                webshops_preds = np.delete(webshops_preds, indices_to_remove, axis=0)
            if len(seq_preds) == 0:
                break

            for d_ix in sorted(indices_to_remove, reverse=True):
                del descs[d_ix]

        # Append predictions which comprise "sentence_length" words
        if len(seq_preds) != 0:
            for ix, seq in enumerate(seq_preds):
                x = [self.ixtoword[str(w)] for w in seq[1:self.sentence_length + 1]]
                final_ypreds.append(x)
                final_yhats.append([descs[ix]])

        return final_yhats, final_ypreds

    def __make_beam_prediction(self, path, beam_size=3):
        """ Method which makes a prediction use beam search for a single path. Extending this to batches greatly
        increases indexing complexity and chance on errors. Since previous tests showed that the BLEU score increase
        is not that high nor differs a lot per model, I have not included this yet. It will just be used for the best
        models to see whether results improve further.
        :param path: path to make beam predictions for
        :param beam_size:
        :return:
        """

        original_beam_size = beam_size
        with open(path, "rb") as f:
            feature = pickle.load(f)
        end_token_ix = self.wordtoix["endseq"]
        desc_name = os.path.basename(path)[:self.desc_filename_length] + ".txt"
        path_to_desc = os.path.join(self.desc_path, desc_name)
        finished_candidates = []

        desc = get_desc(path_to_desc, prediction=True)
        reduce_beam_size = False
        start_seq = [self.wordtoix["startseq"]]

        # Append to list the sequence and the score
        current_candidates = [[start_seq, 0.0]]
        while len(current_candidates[0][0]) < self.sentence_length:
            # allows for batch prediction; which will be the bottleneck. The following greatly reduces computation time

            input_seqs = [sequence.pad_sequences([seq[0]], maxlen=self.sentence_length, padding='post')
                          for seq in current_candidates]
            features = [feature for i in range(len(input_seqs))]
            features = np.array(features).squeeze(axis=1)
            # remove unnecessary dimension in between
            input_seqs = np.array(input_seqs).squeeze(axis=1)
            predictions = self.model.predict([features, input_seqs], batch_size=beam_size ** 2)

            new_candidates = []

            for i, candidate in enumerate(predictions):

                # Get top beam_size predictions and create a new list for the next step. Pred returns indices which
                # correspond to the actual words
                pred = np.argsort(candidate)[-beam_size:]

                # reduce beam_size by 1 if endseq is the best possible word.
                if pred[-1] == end_token_ix:
                    reduce_beam_size = True

                for candidate_word in pred:
                    next_cap, prob = candidate_word, candidate[candidate_word]
                    new_candidate = current_candidates[i][0].copy()
                    new_candidate.append(next_cap)
                    new_prob = math.log(prob) + current_candidates[i][1]
                    if next_cap == end_token_ix:
                        finished_candidates.append([new_candidate, new_prob, new_prob / len(new_candidate)])
                    else:
                        new_candidates.append([new_candidate, new_prob])

            # Retrieve top beam_size scores
            current_candidates = sorted(new_candidates, reverse=True, key=lambda l: l[1])
            current_candidates = current_candidates[:original_beam_size]
            if reduce_beam_size:
                beam_size -= 1
                if beam_size == 0:
                    break

        # Also consider any candidates which ran for the whole sentence length or which were present when beam size
        # went to 0:
        for candidate in current_candidates:
            finished_candidates.append([candidate[0], candidate[1], candidate[1] / len(candidate[0])])

        # use weighted prob
        finished_candidates = sorted(finished_candidates, reverse=True, key=lambda l: l[2])
        finished_candidates_sent = []
        for candidate in finished_candidates:
            sentence = [self.ixtoword[str(w)] for w in candidate[0]]
            finished_candidates_sent.append([sentence, float(candidate[1]), float(candidate[2])])
        best_cap = finished_candidates[0][0]
        best_cap = [self.ixtoword[str(w)] for w in best_cap]

        return best_cap, desc, finished_candidates_sent


class CategoryModel:
    def __init__(self, config):
        """
        Paramaters
        -----------
        config : dict
             dictionary object with all necessary information.
             Key webshop_name : str
                Foldername of the webshop
             Key ... : int
                ddd
        """
        cwd = os.getcwd()

        # Set attributes
        self.webshop_name = config["webshop_name"]
        self.weight_path = os.path.join(cwd, "models", "Weights", self.webshop_name)
        self.output_path = os.path.join(cwd, "models", "Output", self.webshop_name)
        self.var_path = os.path.join(cwd, "variables", self.webshop_name)

        self.img_path = os.path.join(cwd, "Datasets", self.webshop_name, config["img_folder"])
        desc_folder = os.path.join(cwd, "Datasets", self.webshop_name, "Descriptions")
        if not os.path.isdir(desc_folder):
            self.desc_path = os.path.join(cwd, "Datasets", self.webshop_name, config["desc_folder"])
        else:
            self.desc_path = desc_folder

        self.embedding_weights_path = os.path.join(self.var_path, config["embedding_name"])
        self.trainable_embedding = False  # Can be overruled if provided in the config
        self.sentence_length = config["sentence_length"]
        self.nr_nodes = config["nr_nodes"]
        self.output_name = config["output_name"]
        self.dropout = config["dropout"]
        self.desc_filename_length = config["desc_filename_length"]
        self.threshold = config["threshold"]
        self.optimizer = config["optimizer"]
        self.architecture_type = config["architecture_type"]
        self.image_input = config["image_input"] if config["image_input"] else False
        try:
            self.feat_extractor = config["feature_extractor"]
        except KeyError:
            self.feat_extractor = "incv3"
        self.img_dim = config["img_dim"]
        self.loss_function = "categorical_crossentropy"
        self.current_epoch = 0
        self.history = []
        self.val_results = []
        self.evaluations = []
        self.predictions = []
        self.check_print = None
        self.model = None
        self.attribute_included = False
        self.cat_embedding = config["category_embedding"]
        if self.architecture_type == "merge":
            self.concat_add = config["concat_add"]

        # Set optional variables; also makes it possible to override information above if needed
        necessary_key_list = ["sentence_length", "img_folder", "webshop_name", "embedding_name",
                              "nr_nodes", "output_name", "dropout", "epoch_start", "desc_filename_length",
                              "optimizer", "architecture_type"]

        # *args
        remaining_keys = [key for key in config.keys() if key not in necessary_key_list]
        for key in remaining_keys:
            setattr(self, key, config[key])

        try:
            self.feat_dims = tuple(self.feat_dims)
        except AttributeError:
            self.feat_dims = None

        # Load external variables
        if self.embedding_weights_path.endswith(".txt"):
            with open(self.embedding_weights_path, "r", encoding="utf8") as f:
                self.embedding_matrix = np.loadtxt(f)
        elif self.embedding_weights_path.endswith(".p"):
            with open(self.embedding_weights_path, "rb") as f:
                self.embedding_matrix = pickle.load(f)
        else:
            raise ValueError(f"This file extension is currently not supported. Please choose '.p' or '.txt'")

        with open(os.path.join(self.var_path, f"wordtoix_thr{self.threshold}.json"), "r", encoding="utf8") as f:
            self.wordtoix = json.load(f)

        with open(os.path.join(self.var_path, f"ixtoword_thr{self.threshold}.json"), "r", encoding="utf8") as f:
            self.ixtoword = json.load(f)

        # Define vocab size
        self.vocab_size = len(self.wordtoix) + 1

        # Infer partitions from folder structure if explicit paths are not passed in config file
        if "train_imgs" not in config.keys():
            if self.attribute_included == False:
                main_img_path = os.path.join(cwd, "Datasets", self.webshop_name, "resized_imgs")
            else:
                main_img_path = os.path.join(cwd, "Datasets", self.webshop_name, "cropped_images")
            self.train_imgs = []
            self.test_imgs = []
            self.val_imgs = []

            for cat in os.listdir(os.path.join(main_img_path, "TRAIN")):
                for img in os.listdir(os.path.join(main_img_path, "TRAIN", cat)):
                    self.train_imgs.append(img[:-4] + ".p")

            for cat in os.listdir(os.path.join(main_img_path, "VAL")):
                for img in os.listdir(os.path.join(main_img_path, "VAL", cat)):
                    self.val_imgs.append(img[:-4] + ".p")

            for cat in os.listdir(os.path.join(main_img_path, "TEST")):
                for img in os.listdir(os.path.join(main_img_path, "TEST", cat)):
                    self.test_imgs.append(img[:-4] + ".p")

            self.train_imgs = [os.path.join(self.img_path, img) for img in self.train_imgs]
            self.val_imgs = [os.path.join(self.img_path, img) for img in self.val_imgs]
            self.test_imgs = [os.path.join(self.img_path, img) for img in self.test_imgs]
        else:
            self.train_imgs = config["train_imgs"]
            self.val_imgs = config["val_imgs"]
            self.test_imgs = config["test_imgs"]

        # check nr_categories and build category dict
        # first, we build a dictionary which is able to match an image to a category. This way, we are able to match
        # image paths to categories.
        if self.attribute_included == False:
            self.category_dict = {}
            self.categories = []
            for partition in os.listdir(main_img_path):
                for category in os.listdir(os.path.join(main_img_path, partition)):
                    if category not in self.categories:
                        self.categories.append(category)
                    for image in os.listdir(os.path.join(main_img_path, partition, category)):
                        # exclude ".jpg"
                        img_name = image[:-4]
                        self.category_dict[img_name] = category

            # Next, we build a categorical dictionary to use later.
            self.int_dict = {}
            current_index = 0
            for cat in self.categories:
                self.int_dict[cat] = current_index
                current_index += 1
            self.nr_cats = len(self.categories)
        else:
            ix = 0
            with open(os.path.join(cwd, "variables", self.webshop_name, "attribute_dict.json"), 'r') as f:
                attribute_dict = json.load(f)
            self.cat_dict = {}
            for i in sorted(list(attribute_dict.keys())):
                self.cat_dict[i] = ix
                ix += 1
            self.nr_cats = len(self.cat_dict)

    def build_model(self, save_plot=False):
        """

        :param save_plot:
        :return:
        """
        if not self.image_input:
            if self.architecture_type == "merge":
                self.model = build_category_merge_model(self.nr_cats, self.sentence_length, self.vocab_size,
                                                        self.embedding_matrix, self.embedding_matrix.shape[1],
                                                        self.loss_function,
                                                        self.optimizer, self.nr_nodes, concat_add=self.concat_add,
                                                        dropout=self.dropout, cat_embedding=self.cat_embedding,
                                                        attribute_included=self.attribute_included)
            elif self.architecture_type == "brownlee":
                self.model = build_category_brownlee_model(self.nr_cats, self.sentence_length, self.vocab_size,
                                                           self.embedding_matrix, self.embedding_matrix.shape[1],
                                                           self.loss_function,
                                                           self.optimizer, self.nr_nodes, concat_add=self.concat_add,
                                                           dropout=self.dropout, cat_embedding=self.cat_embedding,
                                                           attribute_included=self.attribute_included)
            elif self.architecture_type == "parinject":
                self.model = build_category_parinject_model(self.nr_cats, self.sentence_length, self.vocab_size,
                                                            self.embedding_matrix, self.embedding_matrix.shape[1],
                                                            self.loss_function,
                                                            self.optimizer, self.nr_nodes, concat_add=self.concat_add,
                                                            dropout=self.dropout, cat_embedding=self.cat_embedding,
                                                            attribute_included=self.attribute_included)
            else:
                raise ValueError(f"ARCHITECTURE CURRENTLY NOT SUPPORTED. {self.architecture_type} was supplied while either"
                                 "brownlee, merge or parinject is expected.")
        else:
            if self.architecture_type == "merge":
                self.model = build_img_cat_merge_model(self.nr_cats, self.sentence_length, self.vocab_size,
                                                        self.embedding_matrix, self.embedding_matrix.shape[1],
                                                        self.loss_function,
                                                        self.optimizer, self.nr_nodes, concat_add=self.concat_add,
                                                        dropout=self.dropout, cat_embedding=self.cat_embedding,
                                                        attribute_included=self.attribute_included)
            elif self.architecture_type == "parinject":
                self.model = build_img_cat_parinject_model(self.nr_cats, self.sentence_length, self.vocab_size,
                                                           self.embedding_matrix, self.embedding_matrix.shape[1],
                                                           self.loss_function,
                                                           self.optimizer, self.nr_nodes, concat_add=self.concat_add,
                                                           dropout=self.dropout, cat_embedding=self.cat_embedding,
                                                            attribute_included=self.attribute_included)
            elif self.architecture_type == "brownlee":
                self.model = build_img_cat_brownlee_model(self.nr_cats, self.sentence_length, self.vocab_size,
                                                            self.embedding_matrix, self.embedding_matrix.shape[1],
                                                            self.loss_function,
                                                            self.optimizer, self.nr_nodes, concat_add=self.concat_add,
                                                            dropout=self.dropout, cat_embedding=self.cat_embedding,
                                                            attribute_included=self.attribute_included)
        if save_plot:
            print(self.model.summary())
            plot_model(self.model, os.path.join(self.output_path, self.output_name + "_model.png"))

    def load_weights(self, external_weights_path):
        """Loads weights from an external file.
        Should only be used as a starting position. If the model resumes training, weights are automatically loaded
        based on the latest weight instance.
        Attributes
        ----------
        external_weights_path : path or str
            path to external weights file (should be either .h5 or .hdf5 extension)"""
        self.model.load_weights(external_weights_path)

    def train_model(self, batch_size, nr_epochs, k):
        """

        :param batch_size: batch size
        :param nr_epochs : number of epochs to train for
        :return: nothing is returned inherently, different results are saved to files or in the object attributes
        """
        if self.attribute_included == True:
            self.train_generator = AttributeGenerator(self.cat_dict, self.train_imgs, batch_size, self.desc_path,
                                                      self.vocab_size, self.wordtoix, self.sentence_length,
                                                      shuffle=True, predicting=False, image_input=self.image_input,
                                                      txtfilelength=self.desc_filename_length,
                                                      webshop_name=self.webshop_name)
        else:
            self.train_generator = CategoricGenerator(self.category_dict, self.int_dict, self.train_imgs, batch_size,
                                                  self.desc_path, self.vocab_size, self.wordtoix, self.sentence_length,
                                                  shuffle=True, predicting=False,
                                                  txtfilelength=self.desc_filename_length,
                                                  category_embedding=self.cat_embedding, image_input=self.image_input)

        if self.architecture_type == "brownlee":
            queue_size = 2
        else:
            queue_size = 10
        steps_per_epoch = math.floor(len(self.train_imgs) / batch_size)

        # Epoch iteration
        for epoch in range(self.current_epoch, self.current_epoch + nr_epochs):
            print(" CURRENT EPOCH   ", self.current_epoch)
            # Train for one epoch
            print("TRAINING MODEL >>>   ")
            t = time.time()

            history = self.model.fit_generator(self.train_generator, epochs=1, verbose=1,
                                               steps_per_epoch=steps_per_epoch, max_queue_size=queue_size)
            self.current_epoch += 1
            dt = time.time() - t
            print(f"FINISHED EPOCH took {dt} seconds")

            # Apply validation
            print(" Evaluating ON EPOCH END   ")

            es = self.__on_epoch_end(k, batch_size=batch_size, last_epoch=nr_epochs - 1)
            try:
                self.history.append((epoch, k, history.history, self.val_results[-1]))
            except IndexError:
                print("Can't add val results to history..", self.val_results)
                self.history.append((epoch, k, history.history, []))
                del history

            if es:
                break
        print(f"  FINISHED TRAINING after {self.current_epoch} epochs")

        # Delete remaining files
        os.remove(f"testm_{self.output_name}.h5")
        os.remove(f"tmpoptimizerweights_{self.output_name}.p")
        if self.architecture_type == "attention":
            os.remove(f"testi_{self.output_name}.h5")
            os.remove(f"testt_{self.output_name}.h5")

    def reset_training(self):
        """Resets all training progress.
        """
        self.model.save_weights(f"testm_{self.output_name}.h5")
        # Save optimizer state
        symbolic_weights_m = getattr(self.model.optimizer, 'weights')
        weight_values = K.batch_get_value(symbolic_weights_m)
        with open(f"tmpoptimizerweights_{self.output_name}.p", 'wb') as f:
            pickle.dump(weight_values, f)
        del self.model
        K.clear_session()
        gc.collect()
        self.build_model()
        self.model.load_weights(f"testm_{self.output_name}.h5")
        # Load optimizer weights
        with open(f"tmpoptimizerweights_{self.output_name}.p", "rb") as f:
            optimizer_weights = pickle.load(f)
        # self.model.optimizer.set_weights(optimizer_weights)

    def __on_epoch_end(self, k, batch_size, last_epoch):
        # Run validation data set without beam search to get BLEU scores and save weights if necessary
        if ((self.current_epoch - 1) % k == 0) | (self.current_epoch == 1) | (self.current_epoch == last_epoch):
            predictions, references = self.make_predictions(self.val_imgs, batch_size=batch_size)
            for p, r in zip(predictions[-5:], references[-5:]):
                print("PREDS  ", p, "REF  ", r, "   ")
            print("  COMPUTING BLEU  ")
            score_1, score_2, score_3, score_4 = compute_corpusbleu(references, predictions)
            print("  VALIDATION  :  ", [score_1, score_2, score_3, score_4], "  ")
            self.val_results.append([score_1, score_2, score_3, score_4])

            if len(self.val_results) > 1:
                # Check for early stopping
                if self.val_results[-1][0] <= 0.9 * (self.val_results[-2][0]):
                    print(f" STOPPING TRAINING - Performance no longer improved during epoch. Validation "
                          f"performance in epoch {self.current_epoch} was {self.val_results[-1][0]} "
                          f"versus {self.val_results[-2][0]} in epoch {self.current_epoch - k}")
                    early_stopping = True
                    return early_stopping

                # Also stop early if no improvement over the last 1 or 2 epochs
                if (self.val_results[-1][0] <= (self.val_results[-2][0])) & (len(self.val_results) > 2):
                    if self.val_results[-1][0] <= (self.val_results[-3][0]):
                        print(f" STOPPING TRAINING - Performance no longer improved during epoch. Validation "
                              f"performance in epoch {self.current_epoch} was {self.val_results[-1][0]} "
                              f"versus {self.val_results[-3][0]} in epoch {self.current_epoch - k * 2}")
                        early_stopping = True
                        return early_stopping

            self.model.save_weights(os.path.join(self.weight_path, self.output_name + "_best_weights.h5"))

        else:
            self.reset_training()

    def make_predictions(self, img_paths, batch_size):
        batches = [img_paths[x: x + batch_size] for x in range(0, len(img_paths), batch_size)]
        predictions = []
        references = []

        for batch in tqdm(batches):
            descs, seqs = self.__make_greedy_prediction(batch)
            predictions.extend(seqs)
            references.extend(descs)

        return predictions, references

    def __make_greedy_prediction(self, paths_batch):
        """
        using batches greatly reduces time to make predictions for all images. Up to a max of batch_size times. In
        reality this is a bit slower, as not all items are finished at the same time.
        :param paths_batch: paths to images to make greedy predictions for.
        :return: predictions and references
        """
        endseq_ix = self.wordtoix["endseq"]
        descs = []
        categories = []
        images = []
        batch_size = len(paths_batch)
        for ix, path in enumerate(paths_batch):
            # Remove ".p" and put through dictionaries
            basename = os.path.basename(path)[:-2]
            if self.attribute_included == False:
                category = self.int_dict[self.category_dict[basename]]
                categories.append(category)
            else:
                try:
                    att_filename = os.path.join(cwd, "Datasets", self.webshop_name, "attribute_info",
                                                os.path.basename(path)[:-2] + ".json")
                    with open(att_filename, "r") as f:
                        att_dict = json.load(f)
                except FileNotFoundError as e:
                    print(str(e))
                    batch_size -= 1
                    continue

                attributes = sorted(list(att_dict.keys()))
                attribute_info = [self.cat_dict[i] for i in attributes]
                attribute_info = sequence.pad_sequences([attribute_info], maxlen=len(self.cat_dict))[0]
                categories.append(attribute_info)

            if self.image_input:

                with open(path, 'rb') as f:
                    feature = pickle.load(f)
                try:
                    feature = np.reshape(feature, feature.shape[1])
                except IndexError:
                    pass
                images.append(feature)

            desc_name = os.path.basename(path)[:self.desc_filename_length] + ".txt"
            path_to_desc = os.path.join(self.desc_path, desc_name)
            desc = get_desc(path_to_desc, prediction=True)
            descs.append(desc)
        images = np.array(images)
        # One-hot encoded or embedded?
        if self.attribute_included == True:
            categories = np.array(categories)
        elif self.cat_embedding == False:
            categories = to_categorical(categories, num_classes=len(self.int_dict.keys()))
        else:
            categories = [[cat] for cat in categories]
            categories = np.array(categories)

        # Initialize empty lists to story final results in
        final_yhats = []
        final_ypreds = []

        # Make start sequences for predictions
        startseq = [self.wordtoix["startseq"]]
        startseq_pred = sequence.pad_sequences([startseq], maxlen=self.sentence_length, padding='post')
        seq_preds = [startseq_pred for i in range(batch_size)]
        seq_preds = np.array(seq_preds).squeeze(axis=1)
        print(len(categories), len(seq_preds))
        for i in range(self.sentence_length - 1):
            indices_to_remove = []
            if not self.image_input:
                ypreds = self.model.predict([categories, seq_preds], batch_size=len(paths_batch))
            else:
                ypreds = self.model.predict([categories, seq_preds, images], batch_size=len(paths_batch))

            # Get max probabilities; use this as new index
            ypreds = np.argmax(ypreds, axis=-1)

            for ix, prediction in enumerate(ypreds):

                # Add prediction to current sequence
                seq_preds[ix][i + 1] = prediction

                if prediction == endseq_ix:
                    # if endseq is reached; add prediction and ytrue to finished stuff and pop from sequences to predict
                    # transform to sentence
                    # Transform all class labels up and including endseq to words
                    seq = [self.ixtoword[str(w)] for w in seq_preds[ix][1:i + 1]]
                    final_ypreds.append(seq)
                    final_yhats.append([descs[ix]])
                    indices_to_remove.append(ix)

            # Remove all finished descriptions; in reverse order so previous indices are not disrupted
            seq_preds = np.delete(seq_preds, indices_to_remove, axis=0)
            categories = np.delete(categories, indices_to_remove, axis=0)
            if self.image_input:
                images = np.delete(images, indices_to_remove, axis=0)
            if len(seq_preds) == 0:
                break

            for d_ix in sorted(indices_to_remove, reverse=True):
                del descs[d_ix]

        # Append predictions which comprise "sentence_length" words
        if len(seq_preds) != 0:
            for ix, seq in enumerate(seq_preds):
                x = [self.ixtoword[str(w)] for w in seq[1:self.sentence_length + 1]]
                final_ypreds.append(x)
                final_yhats.append([descs[ix]])

        return final_yhats, final_ypreds

    def evaluate_model(self, BLEU=True, ROUGE=True, nr_steps=None, img_list=None, beam=False, batch_size=1):
        """Method to evaluate a model.
        Makes predictions based on the test images, or another image set if specified. Returns a dictionary containing
        results.
        :param beam:
        :param BLEU : bool
            Whether or not to compute BLEU scores (1-4) (default is True)
        :param ROUGE : bool
            Whether or not to compute ROUGE scores (default is True)
        :param nr_steps : int
            Number of test steps; randomly chosen (default is None)
        :param img_list : list(str) or list(path)
            Files to use as test imgs. If not specified the test set is used. (default is None)

        :return: results : dict
            Dictionary containing, BLEU, ROUGE, predictions and references
        """
        candidates = []
        img_ids = []
        if img_list:
            test_imgs = img_list
        else:
            test_imgs = self.test_imgs

        random.seed(0)
        if nr_steps:
            test_imgs = random.sample(test_imgs, k=nr_steps)

        print("MAKING PREDICTIONS...")
        predictions, references = self.make_predictions(test_imgs, batch_size)

        if BLEU:
            print("Computing BLEU...")
            b1, b2, b3, b4 = compute_corpusbleu(references, predictions)
            BLEUS = {"BLEU_1": b1, "BLEU_2": b2, "BLEU_3": b3, "BLEU_4": b4}
        else:
            BLEUS = []

        predictions = [" ".join(pred) for pred in predictions]
        references = [[" ".join(ref[0])] for ref in references]

        if ROUGE:
            print("Computing ROUGE...")
            ROUGES_avg = compute_ROUGE(predictions, references, aggregator="Avg")
            ROUGES_best = compute_ROUGE(predictions, references, aggregator="Best")
            ROUGES = {"Avg": ROUGES_avg, "Best": ROUGES_best}
        else:
            ROUGES = []
        print(BLEUS, ROUGES)
        results = {"BLEU_SCORES": BLEUS, "ROUGE_SCORES": ROUGES, "IMG_IDS": img_ids, "PREDICTIONS": predictions,
                   "REFERENCES": references,
                   "CANDIDATES": candidates}

        return results
