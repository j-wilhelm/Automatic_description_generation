import json
import os
import tensorflow.keras as k
import numpy as np
import math
import random
import string
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import inception_v3, resnet50
import tensorflow.keras as k
import tensorflow as tf
import PIL
import pickle
from tensorflow.keras.callbacks import Callback
from h_evaluate_model import ModelPrediction
from h_utils import get_desc, desc_to_ix

cwd = os.getcwd()


class InitializationNotImplementedError(Exception):
    pass


class BLEU_validation(Callback):
    def __init__(self, list_test_images, img_folder, desc_folder, desc_filename_length,
                 wordtoix, ixtoword, max_words=120, interval=2, method='greedy', prediction_sample=0):
        super(Callback, self).__init__()

        self.interval = interval
        self.list_test_images = list_test_images
        self.img_folder = img_folder
        self.desc_folder = desc_folder
        self.wordtoix = wordtoix
        self.ixtoword = ixtoword
        self.desc_filename_length = desc_filename_length
        self.max_words = max_words
        self.prediction_method = method
        self.prediction_sample = prediction_sample

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            Intermediate_predictions = ModelPrediction(self.list_test_images, self.img_folder, self.desc_folder,
                                                       self.desc_filename_length, self.model, self.wordtoix,
                                                       self.ixtoword,
                                                       self.max_words, self.prediction_method)

            Intermediate_predictions.make_predictions(BLEU=True, ROUGE=False)
            bleus = Intermediate_predictions.bleu_scores
            print("interval evaluation - epoch: {:d} - score: {}".format(epoch, bleus))
            if self.prediction_sample > 0:
                print("Sample of {} predictions: ".format(str(self.prediction_sample)))
            for i in range(self.prediction_sample):
                random_sample = random.randint(0, len(self.list_test_images) - 1)
                print(random_sample)
                print(Intermediate_predictions.predictions[random_sample])
                print(Intermediate_predictions.references[random_sample])


class CustomImageGeneratorFromDir(tf.compat.v2.keras.utils.Sequence):

    def __init__(self, img_folder, filename_dict, label_dict):
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass

    def on_epoch_end(self):
        pass

class AttributeGenerator(tf.compat.v2.keras.utils.Sequence):
    """
        description: the CategoricGenerator is able to yield input and prediction data for a model with two or three inputs; textual
        data, attribute labels, and image data.

        attributes

        methods
    """
    def __init__(self, category_dict, img_files, batch_size, desc_folder, vocab_size, wordtoix,
                 sentence_length: int, shuffle=True, predicting=False, txtfilelength=9, category_embedding=False,
                 image_input=False, webshop_name=False):
        """

        :param category_dict:
        :param int_dict:
        :param img_files:
        :param batch_size:
        :param desc_folder:
        :param vocab_size:
        :param wordtoix:
        :param sentence_length:
        :param shuffle:
        :param predicting:
        :param txtfilelength:
        :param category_embedding:
        :param image_input: Whether image input is added alongside categories
        """
        self.cat_dict = category_dict # category_name --> integer dict to put in vector
        # Now we initialize the common stuff
        self.batch_size = batch_size
        self.img_files = img_files

        random.seed(0)
        random.shuffle(img_files)
        self.img_files = img_files
        self.sentence_length = sentence_length
        self.wordtoix = wordtoix
        self.indexes = np.arange(len(self.img_files))
        self.desc_folder = desc_folder
        self.vocab_size = vocab_size
        self.shuffle = shuffle
        self.predicting = predicting
        if self.predicting:
            self.shuffle = False
        self.txtfilelength = txtfilelength
        self.on_epoch_end()
        self.ytrue = []
        self.category_embedding = category_embedding
        self.image_input = image_input
        self.webshop_name = webshop_name

    def __len__(self):
        nr_batches = math.floor(len(self.img_files) / self.batch_size)
        print(f"FOUND {len(self.img_files)} IMAGES WHICH ARE SPLIT INTO {nr_batches} BATCHES WITH "
              f"BATCH_SIZE {self.batch_size}")
        return nr_batches

    def on_epoch_end(self):
        # Not sure whether this also is implemented at the start of the experiment.
        # If not, where the fuck does it get self.indexes from?
        self.indexes = np.arange(len(self.img_files))
        if self.shuffle:
            random.shuffle(self.indexes)

    def __getitem__(self, idx):
        """Generate one batch of data
        :param idx: index of the batch
        :return: X and y when fitting. X only when predicting"""
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        list_filepaths_tmp = [self.img_files[k] for k in indexes]
        if self.image_input:
            images, list_filepaths_tmp = self.__generate_img_batch(list_filepaths_tmp)
        # Generate data
        categories = self.__generate_cat_batch(list_filepaths_tmp)

        descriptions = self._generate_description_batch(list_filepaths_tmp)

        # Images if necessary
        if self.image_input:
            if not self.predicting:
                x1, x2, x3, y = self.__generate_sequences_byword_img(categories, descriptions, images)
                print(x1.shape, x2.shape, x3.shape, y.shape)
                return (x1, x2, x3), y
            else:
                for desc in descriptions:
                    self.ytrue.append(desc)
                    # If predicting, we supply "startseq" as variable; this is [1] by definition
                    start_sequence = np.array([1] + (self.sentence_length - 1) * [0])
                    start_sequences = np.tile(start_sequence, (len(categories), 1))
                    return (start_sequences, categories, images), 1

        else:
            if not self.predicting:
                x1, x2, y = self.__generate_sequences_byword(categories, descriptions)
                return (x1, x2), y
            else:
                for desc in descriptions:
                    self.ytrue.append(desc)
                    # If predicting, we supply "startseq" as variable; this is [1] by definition
                    start_sequence = np.array([1] + (self.sentence_length - 1) * [0])
                    start_sequences = np.tile(start_sequence, (len(categories), 1))
                    return (start_sequences, categories), 1

    def __generate_sequences_byword_img(self, categories, descriptions, images):
        x1, x2, x3, y = [], [], [], []
        for i, seq in enumerate(descriptions):
            for j in range(1, len(seq)):
                in_seq, out_seq = seq[:j], seq[j]
                # Pad input sequence
                in_seq = k.preprocessing.sequence.pad_sequences([in_seq], maxlen=self.sentence_length)[0]

                # Encode the output sequence
                out_seq = k.utils.to_categorical([out_seq], num_classes=self.vocab_size)[0]

                # Store the results
                try:
                    x1.append(categories[i])
                    x3.append(images[i])
                except IndexError:
                    print("IndexError")
                    print(i, descriptions[i])
                    print(" len cats ", len(categories))
                    print(" len descs ", len(descriptions))
                    print(" len imgs ", len(images))
                    break
                x2.append(in_seq)
                y.append(out_seq)
        x1 = np.array(x1)
        x2 = np.array(x2)
        x3 = np.array(x3)
        y = np.array(y)
        return x1, x2, x3, y

    def __generate_sequences_byword(self, categories, descriptions):
        x1, x2, y = [], [], []
        for i, seq in enumerate(descriptions):
            for j in range(1, len(seq)):
                in_seq, out_seq = seq[:j], seq[j]
                # Pad input sequence
                in_seq = k.preprocessing.sequence.pad_sequences([in_seq], maxlen=self.sentence_length)[0]

                # Encode the output sequence
                out_seq = k.utils.to_categorical([out_seq], num_classes=self.vocab_size)[0]

                # Store the results
                try:
                    x1.append(categories[i])
                except IndexError:
                    print("IndexError")
                    print(i, descriptions[i])
                    print(" len cats ", len(categories))
                    print(" len descs ", len(descriptions))
                    break
                x2.append(in_seq)
                y.append(out_seq)
        x1 = np.array(x1)
        x2 = np.array(x2)
        y = np.array(y)
        return x1, x2, y


    def __generate_img_batch(self, list_filepaths_tmp):
        images = []

        for i, path in enumerate(list_filepaths_tmp):
            # Store sample
            # \\TODO check pre processing steps necessary here
            try:
                with open(path, 'rb') as f:
                    feature = pickle.load(f)
            # Sometimes a file does not exist. In that case, we continue looping and remove the corrupt item
            except OSError:
                list_filepaths_tmp.pop(i)
                print("OS ERROR OCCURED AT: ", path)
                continue
            except EOFError:
                print("EOF ERROR AT: ", path)
                list_filepaths_tmp.pop(i)
                continue
            # Reshape feature

            try:
                feature = np.reshape(feature, feature.shape[1])
            except IndexError:
                pass

            images.append(feature)

        return images, list_filepaths_tmp
    #\\todo Change
    def __generate_cat_batch(self, list_filepaths_tmp):
        categories = []
        for path in list_filepaths_tmp:
            # Remove ".p" and put through dictionaries
            try:
                att_filename = os.path.join(cwd, "Datasets", self.webshop_name, "attribute_info",
                                            os.path.basename(path)[:-2] + ".json")
                with open(att_filename, "r") as f:
                    att_dict = json.load(f)

                attributes = sorted(list(att_dict.keys()))
                attribute_info = [self.cat_dict[i] for i in attributes]
            except FileNotFoundError:
                attribute_info = [0]
            attribute_info = k.preprocessing.sequence.pad_sequences([attribute_info], maxlen=len(self.cat_dict))[0]
            categories.append(attribute_info)

        categories = np.array(categories)
        return categories

    def _generate_description_batch(self, list_filepaths_tmp):
        descriptions = []
        for i, path in enumerate(list_filepaths_tmp):
            # Get path to y file:
            # depends on how the filename is made. Should have made it standard...
            description_file = os.path.basename(path)[:self.txtfilelength] + ".txt"

            description_path = os.path.join(self.desc_folder, description_file)

            desc = get_desc(description_path)
            ix_desc = desc_to_ix(desc, wordtoix=self.wordtoix)
            # Returns indexed description
            descriptions.append(ix_desc)
        return descriptions



class CategoricGenerator(tf.compat.v2.keras.utils.Sequence):
    """
    description: the CategoricGenerator is able to yield input and prediction data for a model with two inputs; textual
    data and a category label.

    attributes

    methods
    """

    def __init__(self, category_dict, int_dict, img_files, batch_size, desc_folder, vocab_size, wordtoix,
                 sentence_length: int, shuffle=True, predicting=False, txtfilelength=9, category_embedding=False,
                 image_input=False):
        """

        :param category_dict:
        :param int_dict:
        :param img_files:
        :param batch_size:
        :param desc_folder:
        :param vocab_size:
        :param wordtoix:
        :param sentence_length:
        :param shuffle:
        :param predicting:
        :param txtfilelength:
        :param category_embedding:
        :param image_input: Whether image input is added alongside categories
        """
        self.category_dict = category_dict
        self.int_dict = int_dict
        # Now we initialize the common stuff
        self.batch_size = batch_size
        self.img_files = img_files

        random.seed(0)
        random.shuffle(img_files)
        self.img_files = img_files
        self.sentence_length = sentence_length
        self.wordtoix = wordtoix
        self.indexes = np.arange(len(self.img_files))
        self.desc_folder = desc_folder
        self.vocab_size = vocab_size
        self.shuffle = shuffle
        self.predicting = predicting
        if self.predicting:
            self.shuffle = False
        self.txtfilelength = txtfilelength
        self.on_epoch_end()
        self.ytrue = []
        self.category_embedding = category_embedding
        self.image_input = image_input

    def __len__(self):
        nr_batches = math.floor(len(self.img_files) / self.batch_size)
        print(f"FOUND {len(self.img_files)} IMAGES WHICH ARE SPLIT INTO {nr_batches} BATCHES WITH "
              f"BATCH_SIZE {self.batch_size}")
        return nr_batches

    def on_epoch_end(self):
        # Not sure whether this also is implemented at the start of the experiment.
        # If not, where the fuck does it get self.indexes from?
        self.indexes = np.arange(len(self.img_files))
        if self.shuffle:
            random.shuffle(self.indexes)

    #\\TODO Add possibility for image as additional input; use additional function
    def __getitem__(self, idx):
        """Generate one batch of data
        :param idx: index of the batch
        :return: X and y when fitting. X only when predicting"""
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        list_filepaths_tmp = [self.img_files[k] for k in indexes]
        if self.image_input:
            images, list_filepaths_tmp = self.__generate_img_batch(list_filepaths_tmp)
        # Generate data
        categories = self.__generate_cat_batch(list_filepaths_tmp)

        descriptions = self._generate_description_batch(list_filepaths_tmp)

        # Images if necessary
        if self.image_input:
            if not self.predicting:
                x1, x2, x3, y = self.__generate_sequences_byword_img(categories, descriptions, images)
                return (x1, x2, x3), y
            else:
                for desc in descriptions:
                    self.ytrue.append(desc)
                    # If predicting, we supply "startseq" as variable; this is [1] by definition
                    start_sequence = np.array([1] + (self.sentence_length - 1) * [0])
                    start_sequences = np.tile(start_sequence, (len(categories), 1))
                    return (start_sequences, categories, images), 1

        else:
            if not self.predicting:
                x1, x2, y = self.__generate_sequences_byword(categories, descriptions)
                return (x1, x2), y
            else:
                for desc in descriptions:
                    self.ytrue.append(desc)
                    # If predicting, we supply "startseq" as variable; this is [1] by definition
                    start_sequence = np.array([1] + (self.sentence_length - 1) * [0])
                    start_sequences = np.tile(start_sequence, (len(categories), 1))
                    return (start_sequences, categories), 1

    def __generate_sequences_byword_img(self, categories, descriptions, images):
        x1, x2, x3, y = [], [], [], []
        for i, seq in enumerate(descriptions):
            for j in range(1, len(seq)):
                in_seq, out_seq = seq[:j], seq[j]
                # Pad input sequence
                in_seq = k.preprocessing.sequence.pad_sequences([in_seq], maxlen=self.sentence_length)[0]

                # Encode the output sequence
                out_seq = k.utils.to_categorical([out_seq], num_classes=self.vocab_size)[0]

                # Store the results
                try:
                    x1.append(categories[i])
                    x3.append(images[i])
                except IndexError:
                    print("IndexError")
                    print(i, descriptions[i])
                    print(" len cats ", len(categories))
                    print(" len descs ", len(descriptions))
                    print(" len imgs ", len(images))
                    break
                x2.append(in_seq)
                y.append(out_seq)
        x1 = np.array(x1)
        x2 = np.array(x2)
        x3 = np.array(x3)
        y = np.array(y)
        return x1, x2, x3, y

    def __generate_sequences_byword(self, categories, descriptions):
        x1, x2, y = [], [], []
        for i, seq in enumerate(descriptions):
            for j in range(1, len(seq)):
                in_seq, out_seq = seq[:j], seq[j]
                # Pad input sequence
                in_seq = k.preprocessing.sequence.pad_sequences([in_seq], maxlen=self.sentence_length)[0]

                # Encode the output sequence
                out_seq = k.utils.to_categorical([out_seq], num_classes=self.vocab_size)[0]

                # Store the results
                try:
                    x1.append(categories[i])
                except IndexError:
                    print("IndexError")
                    print(i, descriptions[i])
                    print(" len cats ", len(categories))
                    print(" len descs ", len(descriptions))
                    break
                x2.append(in_seq)
                y.append(out_seq)
        x1 = np.array(x1)
        x2 = np.array(x2)
        y = np.array(y)
        return x1, x2, y


    def __generate_img_batch(self, list_filepaths_tmp):
        images = []

        for i, path in enumerate(list_filepaths_tmp):
            # Store sample
            # \\TODO check pre processing steps necessary here
            try:
                with open(path, 'rb') as f:
                    feature = pickle.load(f)
            # Sometimes a file does not exist. In that case, we continue looping and remove the corrupt item
            except OSError:
                list_filepaths_tmp.pop(i)
                print("OS ERROR OCCURED AT: ", path)
                continue
            except EOFError:
                print("EOF ERROR AT: ", path)
                list_filepaths_tmp.pop(i)
                continue
            # Reshape feature

            try:
                feature = np.reshape(feature, feature.shape[1])
            except IndexError:
                pass

            images.append(feature)

        return images, list_filepaths_tmp

    def __generate_cat_batch(self, list_filepaths_tmp):
        categories = []
        for path in list_filepaths_tmp:
            # Remove ".p" and put through dictionaries
            basename = os.path.basename(path)[:-2]
            categories.append(self.int_dict[self.category_dict[basename]])

        # If the categories are not embedded, they should be one-hot encoded
        if self.category_embedding == False:
            categories = k.utils.to_categorical([categories], num_classes=len(self.int_dict.keys()))[0]
        else:
            categories = [[cat] for cat in categories]
            categories = np.array(categories)

        return categories

    def _generate_description_batch(self, list_filepaths_tmp):
        descriptions = []
        for i, path in enumerate(list_filepaths_tmp):
            # Get path to y file:
            # depends on how the filename is made. Should have made it standard...
            description_file = os.path.basename(path)[:self.txtfilelength] + ".txt"

            description_path = os.path.join(self.desc_folder, description_file)

            desc = get_desc(description_path)
            ix_desc = desc_to_ix(desc, wordtoix=self.wordtoix)
            # Returns indexed description
            descriptions.append(ix_desc)
        return descriptions


class CustomGenerator(tf.compat.v2.keras.utils.Sequence):
    """description

    attributes

    methods"""

    def __init__(self, img_files, batch_size, desc_folder, vocab_size, wordtoix, sentence_length: int, n_channels: int,
                 img_dim: tuple, shuffle=True, predicting=False, image_encoder=None,
                 txtfilelength=7, generator_type='word', model_type=None, feat_dims=None, check_print=None,
                 webshop_embedded=False, webshop_dict=None):
        """
        img_files: paths to images
        batch_size: number of images per batch
        dim: dimensions to which images needs to be resized (height, width)
        shuffle: whether to shuffle on epoch end"""
        self.batch_size = batch_size
        self.img_files = img_files

        random.seed(0)
        random.shuffle(img_files)
        self.img_files = img_files
        self.n_channels = n_channels
        self.sentence_length = sentence_length
        self.wordtoix = wordtoix
        self.indexes = np.arange(len(self.img_files))
        self.desc_folder = desc_folder
        self.vocab_size = vocab_size
        self.img_dim = img_dim
        self.shuffle = shuffle
        self.predicting = predicting
        if self.predicting:
            self.shuffle = False
        self.img_encoder = image_encoder
        self.txtfilelength = txtfilelength
        self.on_epoch_end()
        self.generator_type = generator_type
        self.model_type = model_type
        self.feat_dims = feat_dims
        self.ytrue = []
        self.check_print = check_print
        self.webshop_embedded = webshop_embedded
        self.webshop_dict = webshop_dict


    def __len__(self):
        nr_batches = math.floor(len(self.img_files) / self.batch_size)
        print(nr_batches)
        return nr_batches

    def on_epoch_end(self):
        # Not sure whether this also is implemented at the start of the experiment.
        # If not, where the fuck does it get self.indexes from?
        self.indexes = np.arange(len(self.img_files))
        if self.shuffle:
            random.shuffle(self.indexes)

    def __getitem__(self, idx):
        """Generate one batch of data
        :param idx: index of the batch
        :return: X and y when fitting. X only when predicting"""
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        list_filepaths_tmp = [self.img_files[k] for k in indexes]
        # Generate data
        images, list_filepaths_tmp = self._generate_image_batch(list_filepaths_tmp)

        descriptions = self._generate_description_batch(list_filepaths_tmp)
        if not self.predicting:
            if self.generator_type == 'word':
                x1, x2, y, x3 = self.__generate_sequences_byword(images, descriptions, list_filepaths_tmp)
            elif self.generator_type == 'sentence':
                x1, x2, y = self.__generate_sequences_bysentence(images, descriptions)
            if self.model_type[:17] == "multiple_webshops":
                return (x3, x2, x1), y
            else:
                return (x1, x2), y
        else:
            for desc in descriptions:
                self.ytrue.append(desc)
            # If predicting, we supply "startseq" as variable; this is [1] by definition
            start_sequence = np.array([1] + (self.sentence_length - 1) * [0])
            start_sequences = np.tile(start_sequence, (len(images), 1))
            images = np.array(images)
            return (start_sequences, images), 1

    def _generate_image_batch(self, list_filepaths_tmp):
        """Generates data containing batch_size images
        :param list_filepaths_tmp: list of filepaths to load
        :return: batch of images"""

        # Initialization
        images = []
        if self.img_encoder is None:
            images, list_filepaths_tmp = self._generate_image_batch_from_image(list_filepaths_tmp)
        else:
            images, list_filepaths_tmp = self._generate_image_batch_from_features(list_filepaths_tmp)

        return images, list_filepaths_tmp

    def _generate_image_batch_from_features(self, list_filepaths_tmp):
        images = []

        for i, path in enumerate(list_filepaths_tmp):
            # Store sample
            # \\TODO check pre processing steps necessary here
            try:
                with open(path, 'rb') as f:
                    feature = pickle.load(f)
            # Sometimes a file does not exist. In that case, we continue looping and remove the corrupt item
            except OSError:
                list_filepaths_tmp.pop(i)
                print("OS ERROR OCCURED AT: ", path)
                continue
            except EOFError:
                print("EOF ERROR AT: ", path)
                list_filepaths_tmp.pop(i)
                continue
            # Reshape feature

            if self.model_type == "attention":
                feature = np.reshape(feature, (self.feat_dims[0] * self.feat_dims[1],
                                               self.feat_dims[2]))
            else:
                try:
                    feature = np.reshape(feature, feature.shape[1])
                except IndexError:
                    pass

            images.append(feature)

        return images, list_filepaths_tmp

    def _generate_image_batch_from_image(self, list_filepaths_tmp):
        images = []

        # Generate data:
        for i, path in enumerate(list_filepaths_tmp):
            # Store sample
            # \\TODO check pre processing steps necessary here
            try:
                img = load_img(path, target_size=self.img_dim)
                img_array = img_to_array(img)
                img_array = img_array / 255
                images.append(img_array)
            # Sometimes a file is corrupt. In that case, we continue looping and remove the corrupt item
            except PIL.UnidentifiedImageError:
                list_filepaths_tmp.pop(i)
                print("UI Error")
                print(len(list_filepaths_tmp))
                continue
        return images, list_filepaths_tmp

    def _generate_description_batch(self, list_filepaths_tmp):
        descriptions = []
        for i, path in enumerate(list_filepaths_tmp):
            # Get path to y file:
            # depends on how the filename is made. Should have made it standard...
            description_file = os.path.basename(path)[:self.txtfilelength] + ".txt"

            description_path = os.path.join(self.desc_folder, description_file)

            desc = get_desc(description_path)
            ix_desc = desc_to_ix(desc, wordtoix=self.wordtoix)
            # Returns embedded description
            descriptions.append(ix_desc)
        return descriptions

    def __generate_sequences_bysentence(self, images, descriptions):
        input_captions = [x[:-1] for x in descriptions]
        output_captions = [x[1:] for x in descriptions]
        # Pad input captions
        input_captions = np.array([x + [0] * (self.sentence_length - len(x)) for x in input_captions])
        y = np.zeros((len(descriptions), self.sentence_length, self.vocab_size))
        for i, seq in enumerate(output_captions):
            for j, word in enumerate(seq):
                y[i, j, word] = 1.0
            for k in range(j + 1, self.sentence_length):
                y[i, k, 0] = 1.0
            # Pad description
        return input_captions, np.array(images), y

    def __generate_sequences_byword(self, images, descriptions, filenames):
        x1, x2, x3, y = [], [], [], []

        for i, seq in enumerate(descriptions):
            if self.model_type[:17] == "multiple_webshops":
                filename = os.path.basename(filenames[i])
                webshop = self.webshop_dict[filename[:2]]
                # one hot encode if webshop will not be embedded
                if self.webshop_embedded == False:
                    webshop_input = k.utils.to_categorical([webshop],
                                                       num_classes=len(self.webshop_dict.keys()))[0]
                else:
                    webshop_input = [webshop]
            else:
                webshop_input = None
            # split sequence into multiple x, y pairs
            for j in range(1, len(seq)):
                in_seq, out_seq = seq[:j], seq[j]
                # Pad input sequence
                in_seq = k.preprocessing.sequence.pad_sequences([in_seq], maxlen=self.sentence_length)[0]

                # Encode the output sequence
                out_seq = k.utils.to_categorical([out_seq], num_classes=self.vocab_size)[0]

                # Store the results
                try:
                    x1.append(images[i])
                except IndexError:
                    print("IndexError")
                    print(i, descriptions[i])
                    print(len(images))
                    print(len(descriptions))
                    break
                x2.append(in_seq)
                x3.append(webshop_input)
                y.append(out_seq)

        return np.array(x1), np.array(x2), np.array(y), np.array(x3)
