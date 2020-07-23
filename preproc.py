import re
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
import unidecode
import nltk


def preprocessing(text):
    nlp = English()
    stemmer = nltk.stem.PorterStemmer()
    html_pattern = "<.*?>"
    special_chars = "[^ A-Za-z]+"
    text = text.lower()
    text = re.sub(html_pattern, "", text)  # remove any html tags
    text = re.sub(special_chars, "", text)
    text = unidecode.unidecode(text)  # remove accented characters
    text = re.sub(special_chars, "", text)  # remove any special characters
    doc = nlp(text)
    # these words express sentiment. so should not be removed from text
    for word in ["no", "not"]:
        nlp.vocab[word].is_stop = False
    tokens = [token.text for token in doc]  # tokenize text
    tokens = [
        word for word in tokens if not nlp.vocab[word].is_stop
    ]  # remove stop words
    tokens = [
        token.lemma_ for token in doc if token.text in tokens
    ]   # lemmatize every word
    tokens = [stemmer.stem(word) for word in tokens]  # stem the words
    tokens = [word for word in tokens if word.isalnum()]

    return list(set(tokens))
