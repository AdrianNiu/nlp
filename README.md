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

Lemmatization:
In NLP, lemmatization is the process of figuring out the root form or root word (most basic form) or lemma of each word in the sentence. Lemmatization is very similar to stemming, where we remove word affixes to get to the base form of a word. The difference is that the root word is always a lexicographically correct word (present in the dictionary), but the root stem may not be so. Thus, the root word, also known as the lemma, will always be present in the dictionary. It uses a knowledge base called WordNet. Because of knowledge, lemmatization can even convert words that are different and cant be solved by stemmers, for example converting “came” to “come”.

StopWords
Words which have little or no significance, especially when constructing meaningful features from text, are known as stopwords or stop words. These are usually words that end up having the maximum frequency if you do a simple term or word frequency in a corpus. Consider words like a, an, the, be etc. These words don’t add any extra information in a sentence.


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

Q: How exactly does NLP technology go through EHRs and find SDOH? And what does it do with the SDOH data?

A: NLP allows providers to unlock SDOH from EHRs to gain a more complete picture of each patient's circumstances. A user can quickly create a query to extract key concepts and relationships from unstructured patient data to identify issues such as social isolation, transport problems and cultural factors that may impact health and outcomes.
