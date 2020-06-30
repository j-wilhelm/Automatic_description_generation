import string
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
import rouge
import tensorflow.keras.backend as K


def masked_categorical_crossentropy(y_true, y_pred):
    mask_value = 0
    y_true_id = K.argmax(y_true)
    mask = K.cast(K.equal(y_true_id, mask_value), K.floatx())
    mask = 1.0 - mask
    loss = K.categorical_crossentropy(y_true, y_pred) * mask

    # take average w.r.t. the number of unmasked entries
    return K.sum(loss) / K.sum(mask)


def get_desc(path, prediction=False):
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

    if not prediction:
        desc_final = 'startseq ' + ' '.join(desc) + ' endseq'
    else:
        desc_final = desc
    return desc_final


def desc_to_ix(description, wordtoix):
    desc_ix = [wordtoix[word] for word in description.split(" ") if word in wordtoix.keys()]

    return desc_ix


def compute_corpusbleu(references, predictions):
    # Compute BLEU scores
    score_1 = corpus_bleu(references, predictions, weights=(1, 0, 0, 0))
    score_2 = corpus_bleu(references, predictions, weights=(0.5, 0.5, 0, 0))
    score_3 = corpus_bleu(references, predictions, weights=(1 / 3, 1 / 3, 1 / 3, 0))
    score_4 = corpus_bleu(references, predictions, weights=(0.25, 0.25, 0.25, 0.25))

    return score_1, score_2, score_3, score_4


def compute_ROUGE(candidates, references, aggregator='Avg'):
    if aggregator == 'Avg':
        apply_avg = True
        apply_best = False
    elif aggregator == 'Best':
        apply_best = True
        apply_avg = False
    else:
        apply_best = False
        apply_avg = False
    evaluator = rouge.Rouge(metrics=["rouge-n", "rouge-l", "rouge-w"], max_n=4, limit_length=False,
                            apply_avg=apply_avg, apply_best=apply_best, alpha=0.5, stemming=True)

    scores = evaluator.get_scores(candidates, references)

    return scores


def greedy_prediction(path, model):
    pass

class Cider:

    def __init__(self, n=4, df="corpus"):
        """Initialize CIDEr scoring
        """
        self.n = n
        self.df_type = df

    def build_corpus(self):
        pass

    def compute_score(self, references, candidates):

        for cand in candidates:
            pass
            # sanity check

    def __compute_single_score(self, reference, candidate):
        score = self.cider_scorer.compute_score()

def novelty(test_set, reference_set, verbose=0):
    nr_unique = 0
    nr_not_unique = 0

    for caption in test_set:
        if caption not in reference_set:
            nr_unique += 1
        else:
            nr_not_unique += 1

    novelty_score = nr_unique / (nr_not_unique + nr_unique)

    if verbose:
        print("NR UNIQUE : ", nr_unique)
        print("NR SEEN IN TRAIN SET : ", nr_not_unique)
        print("FRACTION OF UNIQUE : ", novelty_score)

    return novelty_score


def vocabulary_size(partition, verbose=0):
    vocab = set()

    for caption in partition:
        vocab.update(caption.split(" "))

    if verbose:
        print("VOCAB SIZE : ", len(vocab))

    return len(vocab), vocab


# This one is slightly difficult; we have the raw version first
# However, we generate captions for different images of the same item
# It is alright (or even excellent) if the same caption is generated for
# Each of those images. For this, we need information on which item the caption
# belongs to.
# NOVELTY
def distinct_raw(partition, verbose=0):
    duplicate_dict = {}

    for caption in partition:
        if caption not in duplicate_dict.keys():
            duplicate_dict[caption] = 0
        else:
            duplicate_dict[caption] += 1

    nr_distinct = len([key for key in duplicate_dict.keys() if duplicate_dict[key] == 1])
    duplicate_captions = [key for key in duplicate_dict.keys() if duplicate_dict[key] > 1]
    nr_duplicates_unique = len(duplicate_captions)
    nr_duplicate_total = sum([duplicate_dict[key] for key in duplicate_captions])
    fraction_distinct = nr_distinct / (nr_distinct + nr_duplicate_total)
    if verbose:
        print("NR DISTINCT CAPTIONS : ", nr_distinct)
        print("NR DUPLICATE CAPTIONS : ", nr_duplicate_total)
        print("FRACTION DISTINCT : ", fraction_distinct)


def batch_dict(refs):
    ord_refs = {}
    for ix, i in enumerate(refs):
        if i[0] in ord_refs.keys():
            ord_refs[i[0]].append(ix)
        else:
            ord_refs[i[0]] = [ix]

    batches = [values for _, values in ord_refs.items()]

    return batches


def distinct_complex(partition, batches, verbose=0):
    duplicate_dict = {}
    nr_capts = len(partition)
    within_item_duplicates = 0
    # Split in image batches based on item it belongs to

    for batch in batches:
        incapt_dict = {}

        for ix in batch:
            caption = partition[ix]
            if caption in incapt_dict:
                incapt_dict[caption] += 1
            else:
                incapt_dict[caption] = 1

        # Calculate number of duplicates
        nr_duplicates_initem = sum([incapt_dict[key] for key in incapt_dict if incapt_dict[key] > 1])

        within_item_duplicates += nr_duplicates_initem
        for key, _ in incapt_dict.items():
            if key in duplicate_dict:
                duplicate_dict[key] += 1
            else:
                duplicate_dict[key] = 1

    nr_distinct = len([key for key in duplicate_dict.keys() if duplicate_dict[key] == 1])
    duplicate_captions = [key for key in duplicate_dict.keys() if duplicate_dict[key] > 1]
    nr_duplicate_total = sum([duplicate_dict[key] for key in duplicate_captions])

    fraction_distinct = nr_distinct / (nr_distinct + nr_duplicate_total)

    if verbose:
        print("NR DISTINCT CAPTIONS : ", nr_distinct)
        print("NR DUPLICATE CAPTIONS : ", nr_duplicate_total)
        print("NR WITHIN ITEM DUPLICATES : ", within_item_duplicates)
        print("TOTAL CAPTIONS : ", nr_capts)
        print("FRACTION DISTINCT : ", fraction_distinct)
        print("FRACTION WITHIN ITEM DUPLICATE : ", within_item_duplicates / nr_capts)

    return fraction_distinct, nr_distinct, nr_duplicate_total, within_item_duplicates


def preprocess_refs(references):
    ord_refs = {}
    for ix, i in enumerate(references):
        if i[0] in ord_refs.keys():
            ord_refs[i[0]].append(ix)
        else:
            ord_refs[i[0]] = [ix]

    batches = [values for _, values in ord_refs.items()]
    references = [x[0] for x in references]

    return references, batches


def diversity_measures(predictions, references, reference_partition, verbose=1):
    references, ref_batches = preprocess_refs(references)
    novelty_score = novelty(predictions, reference_partition, verbose=verbose)
    size_p, vocab_p = vocabulary_size(predictions, verbose=verbose)
    fraction_distinct, nr_distinct, nr_duplicate_total, \
    within_item_duplicates = distinct_complex(predictions, ref_batches, verbose=verbose)

    size_t, vocab_t = vocabulary_size(reference_partition, verbose=0)
    size_r, vocab_r = vocabulary_size(references, verbose=0)

    nr_unique_words = 0
    for i in vocab_p:
        if i not in vocab_r:
            nr_unique_words += 1

    vocab_rel_train = size_p / size_t
    vocab_rel_test = size_p / size_r
    vocab_rel_traintest = size_t / size_r

    diversity_dict = {"novelty_score": novelty_score, "fraction_Distinct": fraction_distinct,
                      "nr_distinct": nr_distinct, "nr_duplicate_total": nr_duplicate_total,
                      "within_item_duplicates": within_item_duplicates,
                      "fraction_within_duplicates": within_item_duplicates / len(references),
                      "vocab_size_ref": size_t, "vocab_size_predictions": size_p, "vocab_perc_intrain": vocab_rel_train,
                      "vocab_perc_inref": vocab_rel_test, "vocab_per_test_train": vocab_rel_traintest}

    return diversity_dict

class CiderScorer:
    def __init__(self, test=None, refs=None, n=4, sigma=6.0):
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ctest = []
        self.document_frequency = {}
        self.ref_len = None
        self.cook_append(test, refs)

    def cook_append(self, test, refs):
        if refs is not None:
            self.crefs.append(cook_refs(refs))