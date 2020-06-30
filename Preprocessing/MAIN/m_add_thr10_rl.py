import codecs
import json
import pickle
import os
import string
import sys

import numpy as np

cwd = os.getcwd()
sys.path.append(cwd)

import tqdm


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

    desc_final = 'startseq ' + ' '.join(desc) + ' endseq'
    return desc_final


def add_to_vocab(vocabulary, desc):
    for word in desc.split(' '):
        vocabulary[word] = vocabulary.get(word, 0) + 1
    return vocabulary


def create_unique_vocab(vocabulary, threshold):
    unique_vocab = {word: vocabulary[word] for word in vocabulary.keys() if vocabulary[word]
                    >= threshold_value}
    return unique_vocab


# for folder in ["Urban_Outfitters", "RALPH_LAUREN"]:
#     var_path = os.path.join(cwd, "variables", folder)
#     with open(os.path.join(var_path, "dataset_class.p"), 'rb') as f:
#         dc = pickle.load(f)
#     dc.vocab_options = {"threshold": 10}
#     print("building_vocab")
#     # Rebuild vocabulary
#     dc.build_vocabulary()
#
#     # Build embedding
#     dc.get_embeddings()

# Redo the
for webshop_name in ["RALPH_LAUREN", "Urban_Outfitters"]:
    var_path = os.path.join(cwd, "variables", webshop_name)
    desc_path = os.path.join(cwd, "Datasets", webshop_name, "Descriptions")
    train_imgs = []
    main_img_path = os.path.join(cwd, "Datasets", webshop_name, "resized_imgs")
    for cat in os.listdir(os.path.join(main_img_path, "TRAIN")):
        for img in os.listdir(os.path.join(main_img_path, "TRAIN", cat)):
            train_imgs.append(img[:-4] + ".p")
    unique_descs = []
    for i in train_imgs:
        if i[:-4] not in unique_descs:
            unique_descs.append(i[:-4] + ".txt")

    train_descs = [os.path.join(desc_path, x) for x in unique_descs]
    for threshold_value in [1, 3]:
        vocabulary = {}
        for path in tqdm.tqdm(train_descs):
            try:
                desc_final = get_desc(path)
            except FileNotFoundError:
                print(path)
                continue

            vocabulary = add_to_vocab(vocabulary, desc_final)

        # Save the vocabulary as a variable in case we want to reuse it
        with open(os.path.join(cwd, "variables", webshop_name, f"full_vocab_{threshold_value}.json"), 'w') as f:
            json.dump(vocabulary, f)

        # Remove words which occur infrequently
        unique_vocab = create_unique_vocab(vocabulary, threshold_value)

        # remove normal vocab, it is no longer needed
        # del self.vocabulary

        # Get the ixtword and wordtoix variables
        ix = 1
        wordtoix = {}
        ixtoword = {}
        for word in unique_vocab:
            wordtoix[word] = ix
            ixtoword[ix] = word
            ix += 1

        vocab_size = len(wordtoix) + 1
        # save these dictionaries
        with open(os.path.join(cwd, "variables", webshop_name,
                               "wordtoix_thr{}.json".format(threshold_value)), 'w') as f:
            json.dump(wordtoix, f)
        with open(os.path.join(cwd, "variables", webshop_name,
                               "ixtoword_thr{}.json".format(threshold_value)), 'w') as f:
            json.dump(ixtoword, f)

        # Get embeddings:
        embedding_dict_glove = {
            "glove_300d_wiki": os.path.join(cwd, "variables", "glove.6B.300d.txt"),
        }

        embedding_dict_fasttext = {
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

            embedding_matrix = np.zeros((vocab_size, embedding_dim))
            print(embedding_matrix.shape)
            # Add the known words to create an embedding matrix which can be used in models
            for word, i in wordtoix.items():
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector

            # Save the resulting embeddings:
            with open(os.path.join(cwd, "variables", webshop_name,
                                   key + "_thr{}_emb.p".format(threshold_value)), 'wb') as f:
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
            embedding_matrix = np.zeros((vocab_size, embedding_dim))
            iter_i = 0
            for word, i in tqdm.tqdm((wordtoix.items())):
                iter_i += 1
                if iter_i % 500 == 0:
                    print("  Processed 500 words... currently at word: {}    ".format(word))
                embedding_vector = embeddings_index.get(word)
                if (embedding_vector is not None) and (len(embedding_vector) > 0):
                    # unfound words will be all zeroes
                    embedding_matrix[i] = embedding_vector

                with open(os.path.join(cwd, "variables", webshop_name,
                                       key + "_thr{}_emb.p".format(threshold_value)), 'wb') as f:
                    pickle.dump(embedding_matrix, f)
