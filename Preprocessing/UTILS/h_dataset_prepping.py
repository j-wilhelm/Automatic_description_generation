import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)

from h_prep_descriptions import get_folders_mp, process_descriptions_from_path_mp
import json
import multiprocessing as mp
import tqdm
from functools import partial
import random
import numpy as np
import string
import codecs
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.applications import inception_v3, resnet50
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow.keras.backend as K
import PIL
from PIL import Image
import pickle
from sklearn.metrics import confusion_matrix


class DataSet_Prepping:
    """A class for complete processing of a new dataset.
    It provides both image and text processing and is optimized for CPU and GPU where relevant.
    Needs a specific folder structure:
    -cwd
    --Datasets
    ---webshop_name
    ----raw_imgs_folder
    ----raw_anns_folder
    --variables
    --models
    ---Weights
    ----webshop_name
    ---Output
    ----webshop_name

    All other folders are made automatically.

    Attributes
    ----------
    raw_imgs_folder: path or str
        Path to folder where raw images are stored
    raw_anns_folder : path or str
        Path to folder where annotation files are stored. Annotation files should contain dicts
        with "description" as mandatory key.
    webshop_name : str
        Name of the webshop as indicated through a folder
    desc_filename_length : int
        Number of characters the base description name comprise. E.g. RL_00001.txt has 8
    train_test_split : tuple
        Percentages for train, val, test partitions.

    description_folder : path or str
        Path to description folder

    feat_extraction_dict : dict
        Dictionary to store process of finetuning extractors

    extracted_features : list
        List to store process of extracting features

    classifier_results : dict
        Dictionary where classifier evaluations are stored in



    Methods
    ------------
    _get_descriptions
    """

    def __init__(self, raw_imgs_folder, raw_anns_folder, webshop_name, desc_filename_length,
                 train_test_split=(0.75, 0.05, 0.2)):
        """
        Initialization: takes the raw iamges and raw annotations. Extract descriptions and splits the data in partitions
        """
        print("NR available CPUs  {}   ".format(mp.cpu_count()))
        print("the raw anns folder is: ")
        print(raw_anns_folder)
        self.webshop_name = webshop_name
        self.raw_imgs_folder = os.path.join(cwd, "Datasets", self.webshop_name, raw_imgs_folder)
        self.raw_anns_folder = os.path.join(cwd, "Datasets", self.webshop_name, raw_anns_folder)
        print("the full path to the anns folder is: ")
        print(self.raw_anns_folder)
        print("                       ")
        self.split_tuple = train_test_split
        self.description_folder = os.path.join(cwd, 'Datasets', self.webshop_name, "Descriptions")
        self.desc_filename_length = desc_filename_length
        self.classifier_results = {}
        self.extracted_features = []
        # Make a variable to store progress and final weights in
        self.feat_extraction_dict = {}
        self.vocab_options = None
        # Make relevant folders if they do not exist
        if not os.path.isdir(self.description_folder):
            os.mkdir(self.description_folder)

        if not os.path.isdir(os.path.join(cwd, "variables", self.webshop_name)):
            os.mkdir(os.path.join(cwd, "variables", self.webshop_name))
            os.mkdir(os.path.join(cwd, "models", "Output", self.webshop_name))
            os.mkdir(os.path.join(cwd, "models", "Weights", self.webshop_name))

        self.output_path = os.path.join(cwd, "models", "Output", self.webshop_name)
        self.weight_path = os.path.join(cwd, "models", "Weights", self.webshop_name)
        # Get the image folders
        self.train_folder = os.path.join(cwd, "Datasets", self.webshop_name, "resized_imgs", "TRAIN")
        self.val_folder = os.path.join(cwd, "Datasets", self.webshop_name, "resized_imgs", "VAL")
        self.test_folder = os.path.join(cwd, "Datasets", self.webshop_name, "resized_imgs", "TEST")
        self.batch_size = 64
        # Get the descriptions, split the data, and build the vocabulary
        self._get_descriptions()
        self.train_descs, self.val_descs, self.test_descs = self._train_test_split()

    def _get_descriptions(self):
        desc_paths, self.filenames_dict = get_folders_mp([self.raw_anns_folder], self.webshop_name)

        filename_dict_path = os.path.join(cwd, "variables", self.webshop_name, "filenames_dict.json")
        with open(filename_dict_path, 'w', encoding='utf8') as f:
            json.dump(self.filenames_dict, f)

        # Use multiprocessing to get descriptions
        nr_processors = mp.cpu_count()
        pool = mp.Pool(nr_processors)
        for _ in tqdm.tqdm(pool.imap_unordered(partial(process_descriptions_from_path_mp,
                                                       output_folder=self.description_folder,
                                                       filenames_dict=self.filenames_dict), desc_paths),
                           total=len(desc_paths)):
            pass

    def build_vocabulary(self):
        if self.vocab_options:
            self.threshold_value = self.vocab_options['threshold']
        # Else set default values
        else:
            self.threshold_value = 5


        print("building a vocabulary...")
        self.vocabulary = {}

        # Add each description in the train dataset to the vocabulary
        for path in tqdm.tqdm(self.train_descs):
            try:
                desc_final = get_desc(path)
            except FileNotFoundError:
                continue

            self.__add_to_vocab(desc_final)

        # Save the vocabulary as a variable in case we want to reuse it
        with open(os.path.join(cwd, "variables", self.webshop_name, f"full_vocab_{self.threshold_value}.json"), 'w') as f:
            json.dump(self.vocabulary, f)

        # Remove words which occur infrequently
        self.__create_unique_vocab(self.threshold_value)

        # remove normal vocab, it is no longer needed
        # del self.vocabulary

        # Get the ixtword and wordtoix variables
        ix = 1
        self.wordtoix = {}
        self.ixtoword = {}
        for word in self.unique_vocab:
            self.wordtoix[word] = ix
            self.ixtoword[ix] = word
            ix += 1

        # save these dictionaries
        with open(os.path.join(cwd, "variables", self.webshop_name,
                               "wordtoix_thr{}.json".format(self.threshold_value)), 'w') as f:
            json.dump(self.wordtoix, f)
        with open(os.path.join(cwd, "variables", self.webshop_name,
                               "ixtoword_thr{}.json".format(self.threshold_value)), 'w') as f:
            json.dump(self.ixtoword, f)

        # Delete the unique vocab, it is no longer needed
        # del self.unique_vocab

        self.vocab_size = len(self.wordtoix) + 1

    def __add_to_vocab(self, desc):
        for word in desc.split(' '):
            self.vocabulary[word] = self.vocabulary.get(word, 0) + 1

    def __create_unique_vocab(self, threshold_value):
        self.unique_vocab = {word: self.vocabulary[word] for word in self.vocabulary.keys() if self.vocabulary[word]
                             >= threshold_value}

    def get_embeddings(self, dicts=None):

        if dicts:
            embedding_dict_glove = dicts[0]
            embedding_dict_fasttext = dicts[1]

        else:
            embedding_dict_glove = {
                "glove_300d_crawl": os.path.join(cwd, "variables", "glove.840B.300d.txt"),
                "glove_300d_wiki": os.path.join(cwd, "variables", "glove.6B.300d.txt"),
                "glove_200d_twitter": os.path.join(cwd, "variables", "glove.6B.200d.txt")
            }

            embedding_dict_fasttext = {
                "fasttext_300d_crawl": os.path.join(cwd, "variables", "fasttext-crawl-300d-2M.vec"),
                "fasttext_300d_wiki": os.path.join(cwd, "variables", "fasttext-wiki-news-300d-1M.vec")
            }

        # Get glove embeddings
        for key, value in embedding_dict_glove.items():
            print(key, value)
            # Initialize the embedding_matrix
            if "300d" in value:
                embedding_dim = 300
            elif "200d" in value:
                embedding_dim = 200
            else:
                raise ValueError("This embedding dimension size is unsupported..")

            print("Starting the embedding of new glove vectors: {}".format(key))
            embeddings_index = {}

            # build the index based on the known words
            f = open(value, encoding='UTF-8')
            for line in f:
                try:
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
                except ValueError as e:
                    print(str(e), "   LINE IS: ", line)

            f.close()

            embedding_matrix = np.zeros((self.vocab_size, embedding_dim))

            # Add the known words to create an embedding matrix which can be used in models
            for word, i in self.wordtoix.items():
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector

            # Save the resulting embeddings:
            with open(os.path.join(cwd, "variables", self.webshop_name,
                                   key + "_thr{}_emb.p".format(self.threshold_value)), 'wb') as f:
                pickle.dump(embedding_matrix, f)

        # Get fasttext embeddings
        for key, value in embedding_dict_fasttext.items():
            print(key, value)
            if "300d" in value:
                embedding_dim = 300
            elif "200d" in value:
                embedding_dim = 200
            else:
                raise ValueError("This embedding dimension size is unsupported..")
            print("Starting the embedding of a new fasttext matrix: {}".format(key))
            # Build embedding index
            embeddings_index = {}
            f = codecs.open(value)
            for line in f:
                try:
                    values = line.rstrip().rsplit(' ')
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
                except ValueError as e:
                    print(str(e), "   LINE IS: ", line)
            f.close()
            print("  Done getting embeddings index..    ")
            # Make embedding matrix to use as layer in future models
            embedding_matrix = np.zeros((self.vocab_size, embedding_dim))
            iter_i = 0
            for word, i in tqdm.tqdm((self.wordtoix.items())):
                iter_i += 1
                if iter_i % 500 == 0:
                    print("  Processed 500 words... currently at word: {}    ".format(word))
                embedding_vector = embeddings_index.get(word)
                if (embedding_vector is not None) and (len(embedding_vector) > 0):
                    # unfound words will be all zeroes
                    embedding_matrix[i] = embedding_vector

                with open(os.path.join(cwd, "variables", self.webshop_name,
                                       key + "_thr{}_emb.p".format(self.threshold_value)), 'wb') as f:
                    pickle.dump(embedding_matrix, f)

    def _train_test_split(self):
        random.seed(0)

        # Shuffle the descriptions
        all_descs = os.listdir(self.description_folder)
        all_desc_files = [os.path.join(self.description_folder, x)
                          for x in all_descs]

        # category_dict = {}
        # keyerrors = 0
        # raw_img_path = r"E:\Jelmer\Uni\Thesis\Data\FARFETCH\IMG"
        # for gender in os.listdir(raw_img_path):
        #     genpath = os.path.join(raw_img_path, gender)
        #     for category in tqdm(os.listdir(genpath)):
        #         catpath = os.path.join(genpath, category)
        #         catname = gender + "_" + category
        #         for img in tqdm(os.scandir(catpath)):
        #             try:
        #                 img_name = img.name[:-4]
        #                 # split img name in parts
        #                 img_parts = img_name.split("_")
        #                 img_id = self.filenames_dict[img_parts[0] + "_" + img_parts[1]]
        #
        #                 category_dict[img_id] = catname
        #             except IndexError:
        #                 if img.name.endswith(".ini"):
        #                     continue
        #             except KeyError:
        #                 keyerrors += 1
        #                 continue
        # print(keyerrors)

        random.shuffle(all_desc_files)
        random.shuffle(all_desc_files)

        # Split into train, val, and test data
        train_split = int(self.split_tuple[0] * len(all_desc_files))
        val_split = int((self.split_tuple[0] + self.split_tuple[1]) * len(all_desc_files))
        train_data = all_desc_files[:train_split]
        val_data = all_desc_files[train_split:val_split]
        test_data = all_desc_files[val_split:]

        return train_data, val_data, test_data

    def image_prepping_cpu(self):
        # Extract and resize train, val and test data
        main_output_path = os.path.join(cwd, "Datasets", self.webshop_name)
        if not os.path.isdir(os.path.join(cwd, "Datasets", self.webshop_name, "resized_imgs")):
            os.mkdir(os.path.join(main_output_path, "resized_imgs"))
            os.mkdir(os.path.join(main_output_path, "resized_imgs", "TRAIN"))
            os.mkdir(os.path.join(main_output_path, "resized_imgs", "VAL"))
            os.mkdir(os.path.join(main_output_path, "resized_imgs", "TEST"))
        # STARTING RESIZING OF IMAGES
        classes = self.split_resize()
        return classes

    def image_prepping(self, training, extractors, testing):
        print(extractors, type(extractors))
        if extractors == "all":
            # Check whether a full restart is needed or not
            ft_extractors, resume_extraction, resume_testing = self.upon_restart()
            print(ft_extractors)
        elif extractors == "full_restart":
            print("    is full restart....     ")
            ft_extractors = "all"
            self.extracted_features = []
            self.classifier_results = {}

        elif type(extractors) == list:
            ft_extractors = extractors
        # Train feature extractors on train folder
        # Do all extractors
        print("  Training extractors...  ")
        if (ft_extractors == "all") & training:
            # Finetune the RL / FF / scratch models further on this task. Use early stopping
            # self.finetune_resnet50("fromscratch")
            # # Clear session in between in order to reset layer numbering. Otherwise extracting specific
            # # layers no longer works
            # K.clear_session()
            # self.finetune_resnet50("fromRL")
            K.clear_session()
            # self.finetune_resnet50("from_FF")
            self.finetune_incv3("fromscratch")
            K.clear_session()
            self.finetune_incv3("fromFF")
            K.clear_session()
            # self.finetune_incv3("from_FF")

        # Or only do specified extractors
        elif training:
            print(" TRAINING YES OR NO ", training)
            for extractor in ft_extractors:
                print(extractor)
                extractor_name = extractor[0]
                print("     ", extractor_name)
                extractor_mode = extractor[1]
                if extractor_name.lower() == "resnet50":
                    self.finetune_resnet50(extractor_mode)
                elif extractor_name.lower() == "incv3":
                    self.finetune_incv3(extractor_mode)
                else:
                    print("THIS EXTRACTOR IS NOT SUPPORTED. CONTINUING...")
                    continue
                K.clear_session()
        print("  training extractors END  ")
        # Get list of all images
        all_images = []
        for partition in [self.train_folder, self.val_folder, self.test_folder]:
            for cat in os.listdir(partition):
                cat_path = os.path.join(partition, cat)

                for img in os.listdir(cat_path):
                    all_images.append(os.path.join(cat_path, img))

        # add non-finetuned weights from FF
        self.feat_extraction_dict["incv3_noft"] = os.path.join(cwd, "models", "Weights", "FF", "Incv3",
                                                               "weights_conv6_01_try1.h5")

        print("  extracting features  ")
        print(self.extracted_features)
        # Feature Extraction
        for extractor, weights_path in self.feat_extraction_dict.items():
            extractor_name = extractor.split("_")[0]
            extractor_mode = extractor.split("_")[1]

            if extractor_name == "incv3":
                layer_names = ["GAP_last", "mixed10"]
            elif extractor_name == "resnet50":
                layer_names = ["GAP_last", "conv5_block3_3_conv"]
            else:
                print("problem           ")
                raise ValueError("extractor_name not supported")
            for layer in layer_names:
                # Check whether extraction has been done already
                print(extractor_name + "_" + extractor_mode + "_" + layer)
                if extractor_name + "_" + extractor_mode + "_" + layer in self.extracted_features:
                    continue
                output_path = os.path.join(cwd, "Datasets", self.webshop_name, extractor + "_" + layer)
                if not os.path.isdir(output_path):
                    os.mkdir(output_path)
                print("   Extracting:  ", extractor_name, "  ", extractor_name, "   ", layer, "   ")
                self.extract_features(extractor_name, extractor_mode, all_images, layer, weights_path, output_path)
        print("        extracting features END     ")

        # Test the classifiers
        print("Testing classifiers")
        print(self.feat_extraction_dict)
        for extractor, weights_path in self.feat_extraction_dict.items():
            extractor_name = extractor.split("_")[0]
            extractor_mode = extractor.split("_")[1]
            if extractor_name == "resnet50":
                test_generator = ImageDataGenerator(preprocessing_function=resnet50.preprocess_input)
                test_gen = test_generator.flow_from_directory(self.test_folder, batch_size=self.batch_size,
                                                              class_mode='categorical', target_size=(224, 224),
                                                              shuffle=False)
            elif extractor_name == "incv3":
                if extractor_mode == "noft":
                    continue
                test_generator = ImageDataGenerator(preprocessing_function=inception_v3.preprocess_input)
                test_gen = test_generator.flow_from_directory(self.test_folder, batch_size=self.batch_size,
                                                              class_mode='categorical', target_size=(299, 299),
                                                              shuffle=False)
            else:
                raise ValueError("Wrong extractor name supplied")
            try:
                loss_acc = self.classifier_results[extractor]
            except KeyError:
                loss_acc = test_classifier(extractor_name, test_gen, weights_path, self.nr_classes, testing="basic")

            basic_results, conmat = test_classifier(extractor_name, test_gen, weights_path, self.nr_classes, testing)
            self.classifier_results[extractor] = {}
            self.classifier_results[extractor]["acc"] = basic_results
            self.classifier_results[extractor]["conmat"] = conmat
            self.classifier_results[extractor]["los_acc"] = loss_acc

        with open(os.path.join(self.output_path, "CLASSIFIERS_RESULTS.json"), 'w') as f:
            json.dump(str(self.classifier_results), f)

        print("DONE")

    def finetune_resnet50(self, mode):
        # Finetune with early stopping on validation loss; save model weights after improvement
        model = build_model("resnet50", self.nr_classes, "training")

        # Set trainable layers
        first_trainable_layer_index = model.layers.index(model.get_layer("conv4_block1_1_bn"))
        for layer in model.layers[:first_trainable_layer_index]:
            layer.trainable = False

        # Load weights if necessary
        if mode == "fromRL":
            print("Loading RL descriptions")
            weights_path = os.path.join(cwd, "models", "Weights", "RL_classification", "Resnet50",
                                        "RL_class_resnet_unfr6_fs_epoch150.h5")
            model.load_weights(weights_path, by_name=True)
        elif mode == "fromFF":
            print("Loading FF descriptions")
            weights_path = os.path.join(cwd, "models", "Weights", "FF_classification", "Resnet50",
                                        "FF_class_resnet_unfr6_fs_epoch150.h5")
            model.load_weights(weights_path, by_name=True)
        else:
            print("Starting from scratch")

        train_generator = ImageDataGenerator(preprocessing_function=resnet50.preprocess_input, horizontal_flip=True)
        val_generator = ImageDataGenerator(preprocessing_function=resnet50.preprocess_input)

        train_gen = train_generator.flow_from_directory(self.train_folder, batch_size=self.batch_size,
                                                        target_size=(224, 224), class_mode='categorical')
        val_gen = val_generator.flow_from_directory(self.val_folder, batch_size=self.batch_size,
                                                    target_size=(224, 224), class_mode='categorical')

        for layer in model.layers[:-4]:
            layer.trainable = False
        # Add early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, restore_best_weights=True)

        # Save model weights if improved
        best_model_folder = os.path.join(self.weight_path, "feature_extractors")
        if not os.path.isdir(best_model_folder):
            os.mkdir(best_model_folder)
        best_model_path = os.path.join(best_model_folder, "best_model_resnet50_{}.h5".format(mode))

        print(model.summary)
        # This may result in a werid runtime error at save time
        # mc = ModelCheckpoint(best_model_path, monitor='val_loss', mode='min', save_best_only=True)

        results = []
        model.compile(optimizer=SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True), loss='categorical_crossentropy',
                      metrics=['accuracy'])
        history = model.fit_generator(train_gen, steps_per_epoch=train_gen.samples / train_gen.batch_size,
                                      epochs=50, validation_data=val_gen, verbose=1, callbacks=[es])
        results.append(history.history)
        model.save_weights(best_model_path + "_frozen")
        for layer in model.layers[first_trainable_layer_index:]:
            layer.trainable = True

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3, restore_best_weights=True)

        model.compile(optimizer=SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True),
                      loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit_generator(train_gen, steps_per_epoch=train_gen.samples / train_gen.batch_size,
                                      epochs=250, validation_data=val_gen, verbose=1, callbacks=[es])
        results.append(history.history)
        model.save_weights(best_model_path)
        self.feat_extraction_dict["resnet50_{}".format(mode)] = best_model_path
        output_folder = os.path.join(self.output_path, "feature_extractors")
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

        with open(os.path.join(output_folder, "history_resnet50_{}.p".format(mode)), 'wb') as f:
            pickle.dump(results, f)

    def finetune_incv3(self, mode):
        model = build_model("incv3", self.nr_classes, "training")
        first_trainable_layer_index = model.layers.index(model.get_layer("conv2d_89"))
        for layer in model.layers[:first_trainable_layer_index]:
            layer.trainable = False

        if mode == "fromRL":
            print("Loading RL descriptions")
            weights_path = os.path.join(cwd, "models", "Weights", "RL_classification", "incv3",
                                        "RL_class_incv3_unfr2blocks_epoch146.h5")
            model.load_weights(weights_path, by_name=True)
        elif mode == "fromFF":
            print("Loading FF descriptions")
            weights_path = os.path.join(cwd, "models", "Weights", "FF", "Incv3",
                                        "weights_conv6_01_try1.h5")
            model.load_weights(weights_path, by_name=True)
        else:
            # Freeze everything for first epochs (until ES)
            for layer in model.layers[:-4]:
                layer.trainable = False
            print("Starting from scratch")
        train_generator = ImageDataGenerator(preprocessing_function=inception_v3.preprocess_input,
                                             horizontal_flip=True)
        val_generator = ImageDataGenerator(preprocessing_function=inception_v3.preprocess_input)

        train_gen = train_generator.flow_from_directory(self.train_folder, batch_size=self.batch_size,
                                                        target_size=(224, 224), class_mode='categorical')
        val_gen = val_generator.flow_from_directory(self.val_folder, batch_size=self.batch_size,
                                                    target_size=(224, 224), class_mode='categorical')

        # Add early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3, restore_best_weights=True)

        # Save model weights if improved
        best_model_folder = os.path.join(self.weight_path, "feature_extractors")
        if not os.path.isdir(best_model_folder):
            os.mkdir(best_model_folder)
        best_model_path = os.path.join(best_model_folder, "best_model_incv3_{}.h5".format(mode))

        print(model.summary)
        # mc = ModelCheckpoint(best_model_path, monitor='val_loss', mode='min', save_best_only=True)

        # Train model for max 250 epochs; unless from scratch then first train for max 50 then unfreeze and
        # train for more
        results = []
        model.compile(optimizer=SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True),
                      loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit_generator(train_gen, steps_per_epoch=train_gen.samples / train_gen.batch_size,
                                      epochs=50, validation_data=val_gen, verbose=1, callbacks=[es])
        results.append(history.history)
        model.save_weights(best_model_path + "_frozen")
        for layer in model.layers[first_trainable_layer_index:]:
            layer.trainable = True

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3, restore_best_weights=True)

        model.compile(optimizer=SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True),
                      loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit_generator(train_gen, steps_per_epoch=train_gen.samples / train_gen.batch_size,
                                      epochs=250, validation_data=val_gen, verbose=1, callbacks=[es])
        results.append(history.history)
        model.save_weights(best_model_path)
        self.feat_extraction_dict["incv3_{}".format(mode)] = best_model_path
        # Save progress
        output_folder = os.path.join(self.output_path, "feature_extractors")
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

        with open(os.path.join(output_folder, "history_incv3_{}.p".format(mode)), 'wb') as f:
            pickle.dump(results, f)

    def extract_features(self, architecture_type, mode, images, layer, weights_path, output_folder):
        # Extracts features based on a model and a layer_name
        # Extract both 2048-d vector and convolutional layer (for attention)
        # Build base model
        if architecture_type == "resnet50":
            model = build_model("resnet50", self.nr_classes, "extracting")
        elif architecture_type == "incv3":
            model = build_model("incv3", self.nr_classes, "extracting")

        # Get all images; train, val and test

        # Loop over layer names (convolutional vs 2048 vector)
        model_input = model.input
        model_output = model.get_layer(layer).output

        model = Model(model_input, model_output)
        model.load_weights(weights_path, by_name=True)

        # Extract all features image by image
        for img_path in images:
            filename = os.path.basename(img_path)[:-4] + ".p"
            try:
                if architecture_type == "incv3":
                    img = load_img(img_path, target_size=(299, 299))
                    x = img_to_array(img)
                    x = np.expand_dims(img, axis=0)
                    x = inception_v3.preprocess_input(x)

                elif architecture_type == "resnet50":
                    img = load_img(img_path, target_size=(224, 224))
                    x = img_to_array(img)
                    x = np.expand_dims(img, axis=0)
                    x = resnet50.preprocess_input(x)

            except PIL.UnidentifiedImageError:
                continue
            feature_vector = model.predict(x)
            with open(os.path.join(output_folder, filename), 'wb') as f:
                pickle.dump(feature_vector, f)

        self.extracted_features.append(architecture_type + "_" + mode + "_" + layer)

    def upon_restart(self):
        # We check whether finetuning has already finished
        print(self.feat_extraction_dict)
        if not self.feat_extraction_dict:
            ft_extractors = "all"
        else:
            all_extractors = {"resnet50_fromscratch", "resnet50_fromRL", "incv3_fromscratch", "incv3_fromRL"}
            completed_extractors = []
            for key, value in self.feat_extraction_dict.items():
                completed_extractors.append(key)
            ft_extractors = all_extractors.difference(completed_extractors)
            ft_extractors = [e.split("_") for e in ft_extractors]

        # Check whether extraction needs to take place
        if self.extract_features:
            resume_extraction = True
        else:
            resume_extraction = False

        if self.classifier_results:
            resume_testing = True
        else:
            resume_testing = False

        return ft_extractors, resume_extraction, resume_testing

    def split_resize(self):
        full_cat_paths = []
        classes = []
        main_folders = os.listdir(self.raw_imgs_folder)
        print("                                                                ")
        print(main_folders)
        print("                                                                ")
        # A very ugly way to get all folders and images while knowing which category they belong to
        if ("MEN" in main_folders) or ("WOMEN" in main_folders):
            for gender in main_folders:
                gender_path = os.path.join(self.raw_imgs_folder, gender)
                for cat in os.listdir(gender_path):
                    cat_path = os.path.join(gender_path, cat)
                    full_cat_paths.append(cat_path)
                    if len(os.listdir(cat_path)) > 0:
                        classes.append(cat)
                    else:
                        continue



        else:
            for cat in main_folders:
                print(cat)
                cat_path = os.path.join(self.raw_imgs_folder, cat)
                if len(os.listdir(cat_path)) > 0:
                    classes.append(cat)
                else:
                    continue
                full_cat_paths.append(cat_path)
        full_cat_paths = set(full_cat_paths)

        # Resize using different CPU cores to greatly decrease duration
        nr_processors = mp.cpu_count()

        pool_2 = mp.Pool(nr_processors)
        for _ in tqdm.tqdm(pool_2.imap_unordered(partial(resize_imgs_mp, filenames_dict=self.filenames_dict,
                                                         webshop_name=self.webshop_name, train_descs=self.train_descs,
                                                         val_descs=self.val_descs, test_descs=self.test_descs),
                                                 full_cat_paths), total=len(full_cat_paths)):
            print("part AAAA    ")
            pass
        return classes


def test_classifier(model_type, test_gen, weights_path, nr_classes, testing):
    model = build_model(model_type, nr_classes, "training")
    model.load_weights(weights_path)
    model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
    # Evaluate simplisticly
    if testing != "extensive":
        basic_results = model.evaluate_generator(test_gen)
        conmat = None
    # Or thoroughly, with a confusion matrix as well
    else:

        ypred = model.predict_generator(test_gen)
        ypred = np.argmax(ypred, axis=1)
        ytrue = test_gen.classes
        print("   nr of items: {} and {}            ".format(len(ytrue), len(ypred)))
        # for i in range(len(ytrue)):
        #     print(ytrue[i], ypred[i])

        conmat = confusion_matrix(ytrue, ypred)
        print(conmat)

        # Get accuracy
        basic_results = sum([ypred[i] == ytrue[i] for i in range(len(ytrue))]) / len(ypred)

    return basic_results, conmat


def build_model(model_type, nr_classes, mode):
    if model_type == "incv3":
        base_model = inception_v3.InceptionV3(include_top=False, weights='imagenet')
    elif model_type == "resnet50":
        base_model = resnet50.ResNet50(include_top=False, weights='imagenet')
    else:
        raise ValueError("This model type is not supported: {}".format(model_type))
    x = base_model.output
    model = GlobalAveragePooling2D(name="GAP_last")(x)
    if mode == "training":
        model = Dropout(0.5, name="dropout_top")(model)
        model = Dense(2048, activation='relu', name="dense2048_{}{}".format(model_type, mode))(model)
        model = Dense(nr_classes, activation='softmax', name="{}_dense_prediction".format(nr_classes))(model)
        model = Model(inputs=base_model.input, outputs=model)
    elif mode == "extracting":
        model = Model(inputs=base_model.input, outputs=model)
        pass

    return model


def resize_imgs_mp(cat_path, filenames_dict, webshop_name, train_descs, val_descs, test_descs, basewidth=299,
                   baseheight=299):
    # Only get basenames of each description path
    train_descs = [os.path.basename(x)[:-4] for x in train_descs]
    val_descs = [os.path.basename(x)[:-4] for x in val_descs]
    test_descs = [os.path.basename(x)[:-4] for x in test_descs]

    main_output_path = os.path.join(cwd, "Datasets", webshop_name, "resized_imgs")
    # Gets the category name so we can use this in the new folders as well
    image_paths = os.listdir(cat_path)
    category_name = os.path.basename(os.path.normpath(cat_path))
    # Add male / female part if necessary. --> very ugly way to get the previous folder name
    prev_folder_name = os.path.split(os.path.split(cat_path)[0])[1]
    if prev_folder_name in ["MEN", "WOMEN"]:
        category_name = prev_folder_name + "_" + category_name

    if (not os.path.isdir(os.path.join(main_output_path, "TRAIN", category_name))) & (len(image_paths) > 0):
        os.mkdir(os.path.join(main_output_path, "TRAIN", category_name))
        os.mkdir(os.path.join(main_output_path, "VAL", category_name))
        os.mkdir(os.path.join(main_output_path, "TEST", category_name))

    print("            ")
    print("looping over images")
    print("   so many:   ")
    print(len(os.listdir(cat_path)))
    nr_keyerrors = 0
    nr_missingdescs = 0
    for image in image_paths:
        if image.endswith(".ini"):
            continue
        # THIS PART IS SHADY >>> very dependent on the naming convention
        img_parts = image.split("_")
        # Filter empty strings
        img_parts = [img_part for img_part in img_parts if img_part]
        basename = "_".join(img_parts[:-1])
        img_ID = img_parts[-1][:-4]
        try:
            basename_desc = filenames_dict[basename]
            output_name = basename_desc + "_" + str(img_ID) + ".JPG"
        except KeyError:
            nr_keyerrors += 1
            continue
        if basename_desc in val_descs:
            output_path = os.path.join(main_output_path, "VAL", category_name, output_name)
        elif basename_desc in test_descs:
            output_path = os.path.join(main_output_path, "TEST", category_name, output_name)
        elif basename_desc in train_descs:
            output_path = os.path.join(main_output_path, "TRAIN", category_name, output_name)
        else:
            nr_missingdescs += 1
            continue
        # Resize image
        # Sometimes an image is corrupt, which can cause several exceptions. Exceptions in general are caught
        #
        try:
            im = PIL.Image.open(os.path.join(cat_path, image))
        except Exception as e:
            print("Error while opening images")
            print(str(e))
            continue
        # Check if image isn't too small
        if (im.size[0] < basewidth) | (im.size[1] < basewidth):
            print("img too small, continuing... ")
            continue

        im = im.resize((basewidth, baseheight))
        im.save(output_path, format="JPEG")
    print("   number of key errors:        ", nr_keyerrors, " nr_missings_descs ", nr_missingdescs)


def get_desc(path):
    table = str.maketrans('', '', string.punctuation)
    with open(path, 'r', encoding='utf8') as f:
        desc = f.read()
    # Tokenize
    desc = desc.split()

    # Convert to lower case
    desc = [word.lower() for word in desc]

    # Remove punctuation
    desc = [word.translate(table) for word in desc]

    # Removing hanging letters
    desc = [word for word in desc if len(word) > 1]

    # Remove tokens with numbers in them
    desc = [word for word in desc if word.isalpha()]

    desc = [word for word in desc if "href" not in word]
    desc = [word for word in desc if "https" not in word]
    desc_final = 'startseq ' + ' '.join(desc) + ' endseq'
    return desc_final
