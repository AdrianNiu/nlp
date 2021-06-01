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

train_pos = train[ train['sentiment'] == 'Positive']
train_pos = train_pos['text']
train_neg = train[ train['sentiment'] == 'Negative']
train_neg = train_neg['text']

def wordcloud_draw(data, color = 'black'):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and not word.startswith('#')
                                and word != 'RT'
                            ])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color=color,
                      width=2500,
                      height=2000
                     ).generate(cleaned_word)
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    
print("Positive words")
wordcloud_draw(train_pos,'white')
print("Negative words")
wordcloud_draw(train_neg)

Lemmatization:
In NLP, lemmatization is the process of figuring out the root form or root word (most basic form) or lemma of each word in the sentence. Lemmatization is very similar to stemming, where we remove word affixes to get to the base form of a word. The difference is that the root word is always a lexicographically correct word (present in the dictionary), but the root stem may not be so. Thus, the root word, also known as the lemma, will always be present in the dictionary. It uses a knowledge base called WordNet. Because of knowledge, lemmatization can even convert words that are different and cant be solved by stemmers, for example converting “came” to “come”.

StopWords
Words which have little or no significance, especially when constructing meaningful features from text, are known as stopwords or stop words. These are usually words that end up having the maximum frequency if you do a simple term or word frequency in a corpus. Consider words like a, an, the, be etc. These words don’t add any extra information in a sentence.

n-grams
n-grams is another representation model for simplifying text selection contents. As opposed to the orderless representation of bag of words, n-grams modeling is interested in preserving contiguous sequences of N items from the text selection. It is usually bi-gram and tri-gram models.

Generative
models the actual distribution of each class
full probabilistic model of all variables - specifies a joint probability distribution over observation and label sequences. 
Bayesian network, Naïve Bayes, Hidden Markov Model…

Discriminative
models the decision boundary between the classes
model only for the target variable(s) conditional on the observed variables
Logistic regression, Decision tree, Support vector machine, k nearest neighbor, conditional random field…

![image](https://user-images.githubusercontent.com/49884281/119600356-e34a6a80-bdb4-11eb-88f7-b3fdb5546796.png)


const { NlpManager } = require('node-nlp');
const manager = new NlpManager({ languages: ['en'] });

// Adds the utterances and intents for the NLP
manager.addDocument('en', 'goodbye for now', 'greetings.bye');
manager.addDocument('en', 'bye bye take care', 'greetings.bye');

// Train also the NLG
manager.addAnswer('en', 'greetings.bye', 'Till next time');
manager.addAnswer('en', 'greetings.bye', 'see you soon!');

// Train and save the model.
(async() => {
    await manager.train();
    manager.save();
    const response = await manager.process('en', 'I should go now');
    console.log(response);
})();

NLP allows providers to unlock SDOH from EHRs to gain a more complete picture of each patient's circumstances. A user can quickly create a query to extract key concepts and relationships from unstructured patient data to identify issues such as social isolation, transport problems and cultural factors that may impact health and outcomes.

IBM Watson API
IBM Watson API combines different sophisticated machine learning techniques to enable developers to classify text into various custom categories. It supports multiple languages, such as English, French, Spanish, German, Chinese, etc. With the help of IBM Watson API, you can extract insights from texts, add automation in workflows, enhance search, and understand the sentiment. The main advantage of this API is that it is very easy to use.
Pricing: Firstly, it offers a free 30 days trial IBM cloud account. You can also opt for its paid plans.
Chatbot API
Chatbot API allows you to create intelligent chatbots for any service. It supports Unicode characters, classifies text, multiple languages, etc. It is very easy to use. It helps you to create a chatbot for your web applications.
Pricing: Chatbot API is free for 150 requests per month. You can also opt for its paid version, which starts from $100 to $5,000 per month.
Speech to text API
Speech to text API is used to convert speech to text
Pricing: Speech to text API is free for converting 60 minutes per month. Its paid version starts form $500 to $1,500 per month.
Sentiment Analysis API
Sentiment Analysis API is also called as 'opinion mining' which is used to identify the tone of a user (positive, negative, or neutral)
Pricing: Sentiment Analysis API is free for less than 500 requests per month. Its paid version starts form $19 to $99 per month.
Translation API by SYSTRAN
The Translation API by SYSTRAN is used to translate the text from the source language to the target language. You can use its NLP APIs for language detection, text segmentation, named entity recognition, tokenization, and many other tasks.
Pricing: This API is available for free. But for commercial users, you need to use its paid version.
Text Analysis API by AYLIEN
Text Analysis API by AYLIEN is used to derive meaning and insights from the textual content. It is available for both free as well as paid from$119 per month. It is easy to use.
Pricing: This API is available free for 1,000 hits per day. You can also use its paid version, which starts from $199 to S1, 399 per month.

leverage clustering algorithms because of their unsupervised nature; this is critical because we have no prior knowledge regarding the noise distribution in the labels, requiring a clustering procedure that is independent of the web-scraped labels.

Suicide example: Data Collection — Our journey started from Data Collection via Reddit’s API (which only allows us to get approximately 1000 unique posts per subreddit).

MTK NLP, short for “MediaTek Natural Language Processing,” is an NLP service native to Android phones. While somewhat notorious for taking full location access to some Android phones, Android’s MTK NLP service is not necessarily representative of NLP as a whole. However, the MTK NLP service does allow for certain NLP applications in Android phones, specifically Google Assistant and Google Lookup—two common applications of NLP software!

Note that if you have had problems with MTK NLP on your Android phone, you will not also have problems with NLP software in your business environment.
