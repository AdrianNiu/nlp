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

##### POS tagging ##################

print("======= Original POS:")
pos_tagged = nltk.pos_tag(tokens)
for word,word_cat in pos_tagged:
    print(word + "_" + word_cat)

print("========= Snowball POS:")
pos_tagged = nltk.pos_tag(snowball_tokens)
for word,word_cat in pos_tagged:
    print(word + "_" + word_cat)

print("========= Porter POS:")
pos_tagged = nltk.pos_tag(porter_tokens)
for word,word_cat in pos_tagged:
    print(word + "_" + word_cat)



