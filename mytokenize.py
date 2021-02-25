import nltk

sentence = "All other systems were reviewed and were negative."

####### Tokenization ###########

tokens = nltk.word_tokenize(sentence)
print(tokens)

####### Stemming  #####################

stemmer1 = nltk.SnowballStemmer("english")
stemmer2 = nltk.PorterStemmer()

snowball_tokens = []
porter_tokens = []

for tok in tokens:
    snowball_tokens.append(stemmer1.stem(tok))
    porter_tokens.append(stemmer2.stem(tok))

print("Original: ",tokens)
print("Snowball: ",snowball_tokens)
print("Porter: ",porter_tokens)

