import csv
import random
import argparse
from time import time
from collections import defaultdict, Counter

import nltk
from sklearn.metrics import precision_recall_fscore_support


"""
This script trains and evaluates a baseline feature set and model
for the symbol disambiguation task. The model is the NLTK NaiveBayesClassifier.
The features used are:
    * The symbol to be disambiguated
    * Bag of words using the 2000 most frequent words in the training data.

This script should serve as a base, which you can modify to implement your own
features, models, evaluation functions, etc. Specifically, you'll likely want
to modify the main() and get_features() functions.

Questions and bugs should be sent to Jake Vasilakes (vasil024@umn.edu).
"""

# So this script produces the same result every time.
# ABSOLUTELY DO NOT CHANGE THIS!
random.seed(42)


def main(infile, outfile):
    """
    The driver function. Performs the following steps:
      * Reads the data in infile and separates the labels from the data.
      * Splits the data into training and testing folds using
        5-fold cross validation.
      * Extracts features from the data in each fold.
      * Trains and evaluates Naive Bayes' models on each fold.
      * Prints the precision, recall, and F1 score for each fold, as well as
        the averages across the folds.

    :param str infile: The path to DeidentifiedSymbolDataset.txt
    :param str outfile: Where to save the results of the evaluation.
    """
    data, labels = read_symbol_dataset(infile, shuffle=True, n=None)
    precs = []
    recs = []
    f1s = []
    fold = 1
    print(f"Running training and evaluation on {len(data)} examples.")
    for test_start, test_end in cross_validation_folds(5, len(data)):
        print(f"Fold: {fold}  ", end='', flush=True)
        fold += 1
        test_data = data[test_start:test_end]
        test_labels = labels[test_start:test_end]
        train_data = data[:test_start] + data[test_end:]
        train_labels = labels[:test_start] + labels[test_end:]

        # Change get_features() to implement your own feature functions.
        train_feats, test_feats = get_features(train_data, test_data)

        print("training...", end='', flush=True)
        start = time()
        train_examples = zip(train_feats, train_labels)
        # Change this line to try different models.
        trained_classifier = nltk.NaiveBayesClassifier.train(train_examples)

        end = time()
        train_time = end - start
        print(f"{train_time:.1f} ", end='', flush=True)
        print("evaluating...", end='', flush=True)
        start = time()
        # If you change classifiers, you may need to change this line.
        predictions = trained_classifier.classify_many(test_feats)
        end = time()
        prec, rec, f1 = evaluate(predictions, test_labels)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
        eval_time = end - start
        print(f"{eval_time:.1f} ", end='', flush=True)
        print("done", flush=True)

        del trained_classifier

    summary = results_summary(precs, recs, f1s)
    with open(outfile, 'w') as outF:
        outF.write(summary)


def read_symbol_dataset(infile, shuffle=True, n=None):
    """
    DO NOT MODIFY THIS FUNCTION!

    Reads DeidentifiedSymbolDataset.txt and separates it into
    data and labels.

    :param str infile: DeidentifiedSymbolDataset.txt
    :param bool shuffle: (Default True) If True, randomly shuffle the order
                         of the examples.
    :param int n: (Default None) If an integer is specified, return that
                  many examples, after shuffling (if shuffle is True).
    :returns: data and labels
    :rtype: tuple(list, list)
    """
    data = []
    labels = []
    with open(infile, 'r', errors="ignore") as inF:
        reader = csv.reader(inF, delimiter='|', quoting=csv.QUOTE_NONE)
        for (i, line) in enumerate(reader):
            # data: [symbol, symbol_position, sample]
            data.append([line[0]] + line[2:])
            labels.append(line[1])
    if shuffle is True:
        shuffled = random.sample(list(zip(data, labels)), k=len(data))
        data = [elem[0] for elem in shuffled]
        labels = [elem[1] for elem in shuffled]
    if n is not None:
        data = data[:n]
        labels = labels[:n]
    return data, labels


def get_features(train_data, test_data):
    """
    Wrapper function for applying a number of feature functions to the data
    in the train and test splits.

    :param list examples: List of raw example data.
    :returns: Features for each example.
    :rtype: list
    """
    # Notice we only obtain the vocabularies from the training data.
    # Why shouldn't we get them from the test data too?
    symbols_vocab = {train_ex[0].lower() for train_ex in train_data}
    vocabulary = get_vocabulary(train_data)

    # (train_features, test_features)
    feature_sets = ([], [])
    for (i, example_set) in enumerate([train_data, test_data]):
        for example in example_set:
            # Add new features in this loop.
            symbol = example[0]
            sym = get_symbol_feature(symbol, symbols_vocab)
            document = example[2]
            bow = get_bag_of_words_features(document, vocabulary)
            feat = {**sym, **bow}
            feature_sets[i].append(feat)
    return feature_sets


def get_vocabulary(examples):
    """
    Get the set of unique words from the text data in examples.
    Used by get_bag_of_words_features().

    :param list examples: List of example data, each of which is
                          a list with plain text data at example[2].
    :returns: Set of unique words.
    :rtype: list
    """
    tokens = [word.lower() for example in examples
              for word in nltk.word_tokenize(example[2])]
    vocabulary = nltk.FreqDist(t for t in tokens)
    return list(vocabulary)[:2000]


def get_symbol_feature(example_symbol, all_symbols):
    """
    Feature representation of the symbol we are trying to disambiguate.

    :param str example_symbol: A symbol, e.g. "Plus".
    :param list all_symbols: The set of all unique symbols.
    :returns: Symbol feature representing this example_symbol.
    :rtype: dict
    """
    features = {}
    for sym in all_symbols:
        features[f"symbol({sym})"] = (sym == example_symbol.lower())
    # Unknown symbol. I.e. we didn't see it in the training set.
    features["UNK"] = (example_symbol.lower() in all_symbols)
    return features


def get_bag_of_words_features(document, vocabulary):
    """
    Bag of words representation of the text in the specified document.

    :param str document: Plain text.
    :param list vocabulary: The unique set of words across all documents.
    :returns: Bag of words features for this document.
    :rtype: dict
    """
    document_words = set(nltk.word_tokenize(document.lower()))
    features = {}
    for word in vocabulary:
        features[f"contains({word})"] = (word in document_words)
    return features


def cross_validation_folds(num_folds, data_size):
    """
    DO NOT MODIFY THIS FUNCTION!

    Given the desired number of cross validation folds and a dataset size
    returns a generator of start, end indices for the test data partitions.

    :param int num_folds: An integer >0 specifying the number of folds.
    :param int data_size: The number of examples in the dataset.
    :returns: Generator of start, end index tuples.
    """
    fold_size = data_size // num_folds
    test_start = 0
    test_end = fold_size
    for k in range(num_folds):
        test_start = fold_size * k
        test_end = test_start + fold_size
        if (k + 1) == num_folds:
            test_end = data_size
        yield test_start, test_end


def evaluate(predictions, gold_labels):
    """
    DO NOT MODIFY THIS FUNCTION!

    Given a model's predictions and the gold standard labels,
    compute the precision, recall, and F1 score of the predictions.

    :param list predictions: Predicted labels.
    :param list gold_labels: Gold standard labels.
    :returns: precision, recall, F1
    :rtype: (float, float, float)
    """
    if len(predictions) != len(gold_labels):
        raise ValueError("Number of predictions and gold labels differ.")
    prec, rec, f1, _ = precision_recall_fscore_support(predictions,
                                                       gold_labels,
                                                       average="weighted",
                                                       zero_division=0)
    return prec, rec, f1


def results_summary(precs, recs, f1s):
    """
    Prints a table of precision, recall, and F1 scores for each
    cross validation fold, as well as the average over the folds.

    :param list precs: The precisions for each fold.
    :param list recs: The recalls for each fold.
    :param list f1s: The F1 scores for each fold.
    """
    assert len(precs) == len(recs) == len(f1s)
    n_folds = len(precs)
    folds_strs = [f"Fold {i+1: <3}" for i in range(n_folds)]
    folds_str = ' '.join(f"{fold_str: <10}" for fold_str in folds_strs)
    precs_str = ' '.join(f"{prec: <10.2f}" for prec in precs)
    precs_avg = sum(precs) / len(precs)
    recs_str = ' '.join(f"{rec: <10.2f}" for rec in recs)
    recs_avg = sum(recs) / len(recs)
    f1s_str = ' '.join(f"{f1: <10.2f}" for f1 in f1s)
    f1s_avg = sum(f1s) / len(f1s)
    outstr = ""
    outstr += f"{'': <13} " + folds_str + f"{'Average': <10}\n"
    outstr += f"{'Precision': <15}" + precs_str + f"{precs_avg: <10.2f}\n"
    outstr += f"{'Recall': <15}" + recs_str + f"{recs_avg: <10.2f}\n"
    outstr += f"{'F1 score': <15}" + f1s_str + f"{f1s_avg: <10.2f}\n"
    return outstr


def describe_data(infile):
    """
    Counts and displays the number of senses per symbol in the dataset.
    Run this function using the --describe_data command line option.

    :param str infile: DeidentifiedSymbolDataset.txt
    """
    symbol2senses = defaultdict(list)
    with open(infile, 'r', errors="ignore") as inF:
        reader = csv.reader(inF, delimiter='|', quoting=csv.QUOTE_NONE)
        for (i, line) in enumerate(reader):
            symbol = line[0]
            sense = line[1]
            symbol2senses[symbol].append(sense)

    all_counts = {}
    for (symbol, senses) in symbol2senses.items():
        sense_counts = Counter(senses)
        all_counts[symbol] = sense_counts
    for (symbol, counts) in all_counts.items():
        print(symbol)
        for (sense, count) in counts.most_common():
            print(f"  {sense}: {count}")
        print()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str,
                        help="""The file containing the input data
                                (i.e. DeidentifiedSymbolDataSet.txt)""")
    parser.add_argument("outfile", type=str,
                        help="Where to write the evaluation result.")
    parser.add_argument("--describe_data", action="store_true", default=False,
                        help="""Compute descriptive statistics about
                                the input dataset before running the
                                training/testing pipeline.""")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.describe_data is True:
        describe_data(args.infile)
    main(args.infile, args.outfile)
