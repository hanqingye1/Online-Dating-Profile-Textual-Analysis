import spacy

def process_chunk(texts):
    # Ensure to load the spaCy model inside the function to avoid issues when using multiprocessing
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    # Filter stopwords, punctuations, lemmatize, and convert to lowercase
    spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
    processed_texts = []
    for doc in nlp.pipe(texts, batch_size=20):
        lemmatized_doc = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
        processed_texts.append(lemmatized_doc)
    return processed_texts