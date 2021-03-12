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
