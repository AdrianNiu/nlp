# NLP analysis pipline

Preprocessing >> Tokenization >> Segmentation >> POS Tagging >> Lemmatization >> Parsing


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

