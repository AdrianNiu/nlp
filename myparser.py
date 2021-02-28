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

raw_sentence1 = "He wants to experience the native american culture and traditions"
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


##################################
### Parsing with a CFG ###########
##################################
#www.nltk.org/book/ch08.html

input_pos_tagged = nltk.pos_tag(tokens)

cfg1 = r"""
        X:{<Y>}
        """

cfg2 = r"""
        NP:{<DT>?<JJ>*(<NN>|<NNS>)}
        """

cfg3 = r"""
        S:{<NP><VP>}
        NP:{<DT>?<JJ>*(<NN>|<NNS>)}
        VP:{<VBD>(<VBN>|<JJ>)<.>?}
        VP:{<VP><CC>?<VP>}
        """

Reg_parser = nltk.RegexpParser(cfg3)
# change the cfg# to adjust context free grammar number to test 
Reg_parser.parse(input_pos_tagged)
Output = Reg_parser.parse(input_pos_tagged)
# print(Output)
# Output.draw()
#
# quit()

##################################
### Parsing with a PCFG ###########
##################################

# define the PCFG
pcfg1 = nltk.PCFG.fromstring("""
    S -> NP VP [1.0]
    NP -> DT NNS [0.5] | DT JJ NNS [0.25] | NNS [0.25]
    VP -> VP CC VP [0.1] | VBD JJ [0.2] | VBD [0.5] | VBD VBN [0.2]
    NNS -> 'systems' [0.7] | 'negative' [0.3]
    DT -> 'the' [0.6] | 'a' [0.2] | 'All' [0.2]
    VBD -> 'reviewed' [0.35] | 'were' [0.65]
    VBN -> 'reviewed' [0.65] | 'were' [0.35]
    CC -> 'and' [1.0]
    JJ -> 'other' [0.5] | 'negative' [0.5]
""")

parser = nltk.pchart.InsideChartParser(pcfg1)
# calling the pchart parser which can be used with PCFG
input = tokens[0:len(tokens)-1]

for parse in parser.parse(input):
  print(parse)
  parse.draw()
# The parser with this case only shows 1 parser senario, but other parser relacing pchart can have mutiple result output