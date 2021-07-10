# NLP analysis pipline
Preprocessing >> Tokenization >> Segmentation >> POS Tagging >> Lemmatization >> Parsing

Natural Language Processing (NLP) offers wide-ranging solutions to retrieve and classify data from EHRs. Text mining, which is part of the NLP family, is defined as the analysis of ‘naturally-occurring’ text driven by human specification to achieve one or a range of goals (e.g. information retrieval or artificial intelligence)

##Two main category of parsing
  New start 
Constituency parsing
identify phrases and their hierarchical (is-a) relations (excludes terminal nodes)

Dependency parsing
identify grammatical relations (lots of them) between syntactic units (including terminal nodes)

Supervised learning
Learn from human knowledge (training data)
Unsupervised learning
Learn from unlabeled data
Semi-supervised learning
Learn from incomplete training data, where a portion of the sample inputs are missing the desired output.


##Validation method:
Macroaveraging: Compute performance for each class, then average.
Microaveraging: Collect decisions for all classes, compute contingency table, evaluate.


Three barriers have impeded accurate identification of suicidal risk. 
1. suicidal behavior is relatively rare and predictive models often require large population samples.
2. risk assessment relies heavily on patient self-report, yet patients may be motivated to conceal their suicidal intentions. 
3. prior to suicide attempts, the last point of clinical contact of patients who die by suicide commonly involves providers with varying levels of suicidal-risk assessment training.

Lemmatization:
In NLP, lemmatization is the process of figuring out the root form or root word (most basic form) or lemma of each word in the sentence. Lemmatization is very similar to stemming, where we remove word affixes to get to the base form of a word. The difference is that the root word is always a lexicographically correct word (present in the dictionary), but the root stem may not be so. Thus, the root word, also known as the lemma, will always be present in the dictionary. It uses a knowledge base called WordNet. Because of knowledge, lemmatization can even convert words that are different and cant be solved by stemmers, for example converting “came” to “come”.

StopWords
Words which have little or no significance, especially when constructing meaningful features from text, are known as stopwords or stop words. These are usually words that end up having the maximum frequency if you do a simple term or word frequency in a corpus. Consider words like a, an, the, be etc. These words don’t add any extra information in a sentence.

Sentiment Analysis - The use of Natural Language Processing techniques to extract subjective information from a piece of text. i.e. whether an author is being subjective or objective or even positive or negative. (can also be referred to as Opinion Mining)

n-grams
n-grams is another representation model for simplifying text selection contents. As opposed to the orderless representation of bag of words, n-grams modeling is interested in preserving contiguous sequences of N items from the text selection. It is usually bi-gram.
Generative
models the actual distribution of each class
full probabilistic model of all variables - specifies a joint probability distribution over observation and label sequences. 

Another useful analysis technique is identifying the different parts of speech within a specific text or a sentence. POS tagging results in a list of tuples; each tuple contains the word and its tag. The tag is a description of the word’s part of speech, is it a verb, noun, adjective, etc.
In most applications, we initially use a default tagger to get basic POS tagging that we can then enhance.

New start: social determinant of health on NLP ， employment stability, relationship dyad, access to lethal means

Corpus

In linguistics and NLP, corpus (literally Latin for body) refers to a collection of texts. Such collections may be formed of a single language of texts, or can span multiple languages -- there are numerous reasons for which multilingual corpora (the plural of corpus) may be useful. Corpora may also consist of themed texts (historical, Biblical, etc.). Corpora are generally solely used for statistical linguistic analysis and hypothesis testing.

Tokenization is the process of demarcating and possibly classifying sections of a string of input characters. The resulting tokens are then passed on to some other form of processing. The process can be considered a sub-task of parsing input

An optimization model seeks to find the values of the decision variables that optimize (maximize or minimize) an objective function among the set of all values for the decision variables that satisfy the given constraints. 

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

seqlen = 40
step = seqlen
sentences = []
for i in range(0, len(text) - seqlen - 1, step):
    sentences.append(text[i: i + seqlen + 1])

x = np.zeros((len(sentences), seqlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), seqlen, len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, (char_in, char_out) in enumerate(zip(sentence[:-1], sentence[1:])):
        x[i, t, char_indices[char_in]] = 1
        y[i, t, char_indices[char_out]] = 1
        
pred_path = 'release_data/qqp/predictions/bert'
suite.run_from_file(pred_path, overwrite=True, file_format='binary_conf')
suite.visual_summary_table()

import spacy

# Load the large English NLP model
nlp = spacy.load('en_core_web_lg')

# The text we want to examine
text = """London is the capital and most populous city of England and 
the United Kingdom.  Standing on the River Thames in the south east 
of the island of Great Britain, London has been a major settlement 
for two millennia. It was founded by the Romans, who named it Londinium.
"""

# Parse the text with spaCy. This runs the entire pipeline.
doc = nlp(text)


Systematic review of suicide prediction and VA data access

Dependency Parsing
The next step is to figure out how all the words in our sentence relate to each other. This is called dependency parsing

import numpy as np
countries = ['France', 'Germany', 'Brazil']
for country in countries:
    ts = editor.template('{male} {last} is from {city}',
                male=editor.lexicons.male_from[country],
                last=editor.lexicons.last_from[country],
                city=editor.lexicons.country_city[country],
               )
    print('Country: %s' % country)
    print('\n'.join(np.random.choice(ts.data, 3)))
    print()
    
    
    SPACY
    
    # pip install -U spacy
# python -m spacy download en_core_web_sm
import spacy

# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_sm")

# Process whole documents
text = ("When Sebastian Thrun started working on self-driving cars at "
        "Google in 2007, few people outside of the company took him "
        "seriously. “I can tell you very senior CEOs of major American "
        "car companies would shake my hand and turn away because I wasn’t "
        "worth talking to,” said Thrun, in an interview with Recode earlier "
        "this week.")
doc = nlp(text)

# Analyze syntax
print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

# Find named entities, phrases and concepts
for entity in doc.ents:
    print(entity.text, entity.label_)
    
    running task:
    # One line per example
test.to_raw_file('/tmp/raw_file.txt')
# each line has prediction probabilities (softmax)
test.run_from_file('/tmp/softmax_preds.txt', file_format='softmax', overwrite=True)


# Disadvantages of single feature evaluation
Relevance between features are ignored
Features could be redundant 
A feature that is completely useless by itself can provide a significant performance improvement when taken with others
Two features that are useless by themselves can be useful together

Latent Dirichlet Allocation (LDA) - A common topic modeling technique, LDA is based on the premise that each document or piece of text is a mixture of a small number of topics and that each word in a document is attributable to one of the topics.


# Dimensionality Reduction

Principal component analysis
mathematical procedure that uses orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components.
Keeping only the first L principal components, produced by using only the first L loading vectors, gives the truncated transformation


One example would be to look for patterns from multiple data sources (like patient portal messages, nurse notes, telehealth transcripts and even social media) and analyze the data. Data could be extracted using Natural Language Processing (NLP) via an Augmented Intelligence (AI) process which allows you to access and combine both structured and unstructured text content to get a clear depiction of the potential issues arising. 

python -m venv .env
source .env/bin/activate
pip install -U pip setuptools wheel
pip install -U spacy


NLP SDOH Employment instability

