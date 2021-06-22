# NLP analysis pipline

Preprocessing >> Tokenization >> Segmentation >> POS Tagging >> Lemmatization >> Parsing

Natural Language Processing (NLP) offers wide-ranging solutions to retrieve and classify data from EHRs. Text mining, which is part of the NLP family, is defined as the analysis of ‘naturally-occurring’ text driven by human specification to achieve one or a range of goals (e.g. information retrieval or artificial intelligence)

##Two main category of parsing

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

n-grams
n-grams is another representation model for simplifying text selection contents. As opposed to the orderless representation of bag of words, n-grams modeling is interested in preserving contiguous sequences of N items from the text selection. It is usually bi-gram and tri-gram models.

Name Entity Recognition
NER is an essential NLP task that allows us to spot the main entities in a text. The most popular ways of extracting entities from the text include the Lexicon approach, Rule-based systems, Machine learning-based systems, and Hybrid approach


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
