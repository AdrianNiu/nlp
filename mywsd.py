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


#################################
### WSD ########################
################################

from nltk.wsd import lesk
from nltk.corpus import wordnet as wn

# install wordnet db
nltk.download('wordnet')

raw_sentence1 = "He grows up in Chinese culture and traditions"
raw_sentence2 = "Staph bacteria were found in his culture"

# tokenize
sent1_toks = nltk.word_tokenize(raw_sentence1)
sent2_toks = nltk.word_tokenize(raw_sentence2)

pos_tagged_1 = nltk.pos_tag(sent1_toks)
pos_tagged_2 = nltk.pos_tag(sent2_toks)

pos1 = pos_tagged_1[7][1]
pos2 = pos_tagged_2[6][1]

print(pos_tagged_1)
print(pos_tagged_2)


# use Lesk algorithm to disambiguate
print("Meaning of the word culture in the first sentence: ", lesk(sent1_toks, 'culture','n'))
print("Meaning of the word culture in the second sentence: ", lesk(sent2_toks, 'culture','n'))

# let's examine various sense of the word 'culture'
print("\nSynsets of the word culture in WordNet:")
for ss in wn.synsets('culture'):
    print(ss, ss.definition())


