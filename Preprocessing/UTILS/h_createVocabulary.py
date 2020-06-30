import string
import numpy as np
from tqdm import tqdm


class ImplementationError(Exception):
    pass


def descriptions_to_array(filepaths, embeddings_index, init=True, vocabulary=None, threshold=5):
    """Get all descriptions and compute vocabulary to get embedding matrix"""
    if init & (vocabulary is not None):
        raise ImplementationError("User must provide either a vocabulary or set init to true. Not both.")

    if (init is False) & (vocabulary is None):
        raise ImplementationError("User must provide either a vocabulary or set init to true.")
    if init:
        vocabulary = initialize_vocab()
    all_desc = []
    print("/n")
    print("Building a vocabulary")
    for path in tqdm(filepaths):
        try:
            desc_final = get_desc(path)
        except FileNotFoundError:
            continue

        vocabulary = add_to_vocab(vocabulary=vocabulary, description=desc_final)
        all_desc.append(desc_final)

    vocab = unique_vocab(vocabulary=vocabulary, threshold=threshold)
    print("/n")
    print("Building an embedding matrix")
    ixtoword, wordtoix = encode_words(vocabulary=vocab)
    embedding_matrix = create_embedding_mat(words_dict=wordtoix, embeddings_index=embeddings_index)
    print("/n")
    print("Embedding the descriptions")
    ix_desc = desc_to_ix(descriptions=all_desc, wordtoix=wordtoix)

    return all_desc, ix_desc, vocab, embedding_matrix, wordtoix, ixtoword


def initialize_vocab():
    vocabulary = {}

    return vocabulary


def add_to_vocab(vocabulary, description):
    for word in description.split(' '):
        vocabulary[word] = vocabulary.get(word, 0) + 1
    return vocabulary


def unique_vocab(vocabulary, threshold=10):
    """vocabulary  : dict of all known words in the sample
    threshold : int, default = 10. Minimum number of occurences to be included"""
    vocab = {word: vocabulary[word] for word in vocabulary.keys() if vocabulary[word] >= threshold}
    return vocab


def encode_words(vocabulary):
    ix = 1
    wordtoix = {}
    ixtoword = {}

    for word in vocabulary.keys():
        wordtoix[word] = ix
        ixtoword[ix] = word
        ix += 1

    return ixtoword, wordtoix


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


def desc_to_ix(descriptions, wordtoix):
    idx_descriptions = []
    for desc in tqdm(descriptions):
        desc_ix = [wordtoix[word] for word in desc.split(" ") if word in wordtoix.keys()]
        idx_descriptions.append(desc_ix)

    return idx_descriptions


# Create embedding matrix for all unique words
def create_embedding_mat(words_dict, embeddings_index, embedding_dim=200):
    """
    words_dict: dict, dictionary with word, word_ID pairs
    embedding_dim: int, default = 200, number of dimensions to use"""
    vocab_size = len(words_dict) + 1
    print("Vocab size is: " + str(vocab_size))
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in words_dict.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
