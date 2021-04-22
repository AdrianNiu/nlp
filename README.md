# NLP analysis pipline

Preprocessing >> Tokenization >> Segmentation >> POS Tagging >> Lemmatization >> Parsing

Test Corpus

Lexical sample − This kind of corpora is used in the system, where it is required to disambiguate a small sample of words.

All-words − This kind of corpora is used in the system, where it is expected to disambiguate all the words in a piece of running text.

##Two main category of parsing

Constituency parsing
identify phrases and their hierarchical (is-a) relations (excludes terminal nodes)

Dependency parsing
identify grammatical relations (lots of them) between syntactic units (including terminal nodes)

What Is a Corpus?
A corpus is a collection of machine-readable texts that have been produced in a natural communicative setting. 

Active learning aims at reducing the number of examples required to achieve the desired accuracy by selectively sampling the examples for user to label and train the classifier.

Validation method:
Macroaveraging: Compute performance for each class, then average.
Microaveraging: Collect decisions for all classes, compute contingency table, evaluate.

abbreviation_disambiguation_baseline.py

txt = dtf["text"].iloc[0]
print(txt, " --> ", langdetect.detect(txt))

Naive Bayes

P(c|x) = P(x|c) * P(c) / P(x)

Generative
models the actual distribution of each class
full probabilistic model of all variables - specifies a joint probability distribution over observation and label sequences. 
Bayesian network, Naïve Bayes, Hidden Markov Model…

Discriminative
models the decision boundary between the classes
model only for the target variable(s) conditional on the observed variables
Logistic regression, Decision tree, Support vector machine, k nearest neighbor, conditional random field…

#### SVM
SVM is a discriminative classifier formally defined by a separating hyperplane
Given labeled training data (supervised learning), the algorithm outputs an optimal hyperplane which categorizes new examples
In two dimensional space this hyperplane is a line dividing a plane in two parts where in each class lay in either side

In NLP mental health prediction research, when studying proxy signals, it is crusical to see if it actually mapped to a good measurement of mental illness and symptoms.

Word embeddings help in the following use cases.
Compute similar words
Text classifications
Document clustering/grouping
Feature extraction for text classifications
Natural language processing.

AWS Kinesis works as a temporary storage mechanism for faster retrieval for further downstream components and NLP. Once data is produced, it is consumed by the consumer. 

Bags of words
The most intuitive way to do so is to use a bags of words representation:

Assign a fixed integer id to each word occurring in any document of the training set (for instance by building a dictionary from words to integer indices).

For each document #i, count the number of occurrences of each word w and store it in X[i, j] as the value of feature #j where j is the index of word w in the dictionary.

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
    
Word2Vec
Word2Vec — Word representations in Vector Space founded by Tomas Mikolov and a group of a research team from Google developed this model in 2013.

Standard NLP Workflow
CRISP-DM Model is a Cross-industry standard process for data mining, known as CRISP-DM, which is an open standard process model that describes common approaches used by data mining experts. It is the most widely-used analytics model. Typically, any NLP-based problem can be solved by a methodical workflow that has a sequence of steps. The major steps are depicted in the following figure.

Syntactic Regularities: Refers to grammatical sentence correction.
Semantic Regularities: Refers to the meaning of the vocabulary symbols arranged in that structure.

## for data
import pandas as pd
import collections
import json
## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import wordcloud
## for text processing
import re
import nltk
## for language detection
import langdetect 
## for sentiment
from textblob import TextBlob
## for ner
import spacy
## for vectorizer
from sklearn import feature_extraction, manifold
## for word embedding
import ge
>>> def pos_features(word):
...     features = {}
...     for suffix in common_suffixes:
...         features['endswith({})'.format(suffix)] = word.lower().endswith(suffix)
...     return features


def get_short_form_feature(short_form, all_short_forms):
    """
    Feature representation of the short form we are trying to disambiguate.

    :param str short_form: An abbreviation or acronym, e.g. "AB".
    :param list all_short_forms: The set of all unique abbreviations/acronyms.
    :returns: Feature representing this short form.
    :rtype: dict
    """
    features = {}
    for sf in all_short_forms:
        features[f"short_form({sf})"] = (sf == short_form.lower())
    # Unknown short_form. I.e. we didn't see it in the training set.
    features["UNK"] = (short_form.lower() in all_short_forms)
    return features
    
>>> classifier = nltk.DecisionTreeClassifier.train(train_set)
>>> nltk.classify.accuracy(classifier, test_set)
0.62705121829935351

Convolutional Layer
A convolutional layer can be thought of as composed of a series of “maps” called the “feature map” or the “activation map” . Each activation map has two components :
A linear map, obtained by convolution over maps in the previous layer (each linear map has, associated with it, a learnable filter or kernal)
An activation that operates on the output of the convolution

Training using Prodigy.

bash-3.2$ prodigy dataset spooky ✨ Successfully added ‘spooky’ to database SQLite.

Prodigy requires a jsonl file as input data, with a json dictionary per line.
These json data must have the following parameters as minimal:
{“text”:”text of the sentence”,”label”:”category of the sentence”,”answer”:”’reject’ or ‘accept’”}

spaCy model	One of the available model packages for spaCy. Models can be installed as Python packages and are available in different sizes and for different languages. They can be used as the basis for training your own model with Prodigy.

spaCy is to build information extraction or natural language understanding systems, or to pre-process text for deep learning.

Prodigy recipe is a Python function that can be run via the command line. 

The brat standalone server only is available in brat v1.3 and above.
The standalone server is experimental and should not be used for sensitive data or systems accessible from the internet.

brat is not compatible with Python 3. Thus you might have to modify the command python standalone.py to python2 standalone.py.

Under annotation catagory: text span annotations, such as those marked with the Organization and Person types in the example
relation annotations, such as the Family relation in the example

• Feature: the string representing the input text
• Target: the text’s polarity (0 or 1)

# >> FEATURE SELECTION << #
def remove_correlated_features(X):
    corr_threshold = 0.9
    corr = X.corr()
    drop_columns = np.full(corr.shape[0], False, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if corr.iloc[i, j] >= corr_threshold:
                drop_columns[j] = True
    columns_dropped = X.columns[drop_columns]
    X.drop(columns_dropped, axis=1, inplace=True)
    return columns_dropped
def remove_less_significant_features(X, Y):
    sl = 0.05
    regression_ols = None
    columns_dropped = np.array([])
    for itr in range(0, len(X.columns)):
        regression_ols = sm.OLS(Y, X).fit()
        max_col = regression_ols.pvalues.idxmax()
        max_val = regression_ols.pvalues.max()
        if max_val > sl:
            X.drop(max_col, axis='columns', inplace=True)
            columns_dropped = np.append(columns_dropped, [max_col])
        else:
            break
    regression_ols.summary()
    return columns_dropped
    
    from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC


# Transform each text into a vector of word counts
vectorizer = CountVectorizer(stop_words="english",
                             preprocessor=clean_text)

training_features = vectorizer.fit_transform(train_data["text"])    
test_features = vectorizer.transform(test_data["text"])

# Training
model = LinearSVC()
model.fit(training_features, train_data["sentiment"])
y_pred = model.predict(test_features)

# Evaluation
acc = accuracy_score(test_data["sentiment"], y_pred)

print("Accuracy on the IMDB dataset: {:.2f}".format(acc*100))

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

Tokenization: is used to segment the input text into its constituents words (tokens). In this way, it becomes easier to then convert our data into a numerical format.
Stop Words Removal: is applied in order to remove from our text all the prepositions (eg. “an”, “the”, etc…) which can just be considered as a source of noise in our data (since they do not carry additional informative information in our data).
Stemming: is finally used in order to get rid of all the affixes in our data (eg. prefixes or suffixes). In this way, it can in fact become much easier for our algorithm to not consider as distinguished words which have actually similar meaning (eg. insight ~ insightful).


