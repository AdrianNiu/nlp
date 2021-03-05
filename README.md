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
