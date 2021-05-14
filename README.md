# NLP analysis pipline

Preprocessing >> Tokenization >> Segmentation >> POS Tagging >> Lemmatization >> Parsing

Natural Language Processing (NLP) offers wide-ranging solutions to retrieve and classify data from EHRs. Text mining, which is part of the NLP family, is defined as the analysis of ‘naturally-occurring’ text driven by human specification to achieve one or a range of goals (e.g. information retrieval or artificial intelligence)

##Two main category of parsing

Constituency parsing
identify phrases and their hierarchical (is-a) relations (excludes terminal nodes)

Dependency parsing
identify grammatical relations (lots of them) between syntactic units (including terminal nodes)



##Validation method:
Macroaveraging: Compute performance for each class, then average.
Microaveraging: Collect decisions for all classes, compute contingency table, evaluate.


Three barriers have impeded accurate identification of suicidal risk. 
1. suicidal behavior is relatively rare and predictive models often require large population samples.
2. risk assessment relies heavily on patient self-report, yet patients may be motivated to conceal their suicidal intentions. 
3. prior to suicide attempts, the last point of clinical contact of patients who die by suicide commonly involves providers with varying levels of suicidal-risk assessment training.

def mem_usage(df: pd.DataFrame) -> str: 
"""This method styles the memory usage of a DataFrame to be readable as MB. Parameters ---------- df: pd.DataFrame Data frame to measure. Returns ------- str Complete memory usage as a string formatted for MB. """ 
    return f'{df.memory_usage(deep=True).sum() / 1024 ** 2 : 3.2f} MB'

def convert_df(df: pd.DataFrame, deep_copy: bool = True) -> pd.DataFrame: 
"""Automatically converts columns that are worth stored as ``categorical`` dtype. Parameters ---------- df: pd.DataFrame Data frame to convert. deep_copy: bool Whether or not to perform a deep copy of the original data frame. Returns ------- pd.DataFrame Optimized copy of the input data frame. """ 
    return df.copy(deep=deep_copy).astype({ col: 'category' for col in df.columns if df[col].nunique() / df[col].shape[0] < 0.5})

Lemmatization:
In NLP, lemmatization is the process of figuring out the root form or root word (most basic form) or lemma of each word in the sentence. Lemmatization is very similar to stemming, where we remove word affixes to get to the base form of a word. The difference is that the root word is always a lexicographically correct word (present in the dictionary), but the root stem may not be so. Thus, the root word, also known as the lemma, will always be present in the dictionary. It uses a knowledge base called WordNet. Because of knowledge, lemmatization can even convert words that are different and cant be solved by stemmers, for example converting “came” to “come”.

StopWords
Words which have little or no significance, especially when constructing meaningful features from text, are known as stopwords or stop words. These are usually words that end up having the maximum frequency if you do a simple term or word frequency in a corpus. Consider words like a, an, the, be etc. These words don’t add any extra information in a sentence.

A stall is injected into the pipeline by the processor to resolve data hazards (situations where the data required to process an instruction is not yet available. A NOP is just an instruction with no side-effect.

# importing NLTK library stopwords
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize

## print(stopwords.words('english'))

