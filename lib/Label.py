import re

# try:
#     import spacy
#
#     nlp = spacy.load("en_core_web_sm")
#
#     def word_tokenize(x):
#         doc = nlp(str(x), disable=["tagger", "parser", "ner", "lemmatizer", "textcat"])
#         return [token.text.strip() for token in doc if not token.is_punct]
#
# except ImportError(spacy):
#     print("default nltk tokenize")
from nltk.tokenize import word_tokenize

camelcase = re.compile(r".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)")


def tokenize(item: str) -> str:
    item = (
        item.replace("_", " ")
        .replace("-", " ")
        .replace(",", " ")
        # .replace(".", " ")  # "e.g."
        .replace("/", " ")
        .replace('"', " ")
        .replace("'", " ")
        .replace("\\", " ")
        .replace("(", " ")
        .replace(")", " ")
        .lower()
    )
    words = word_tokenize(item)
    return " ".join(words)


def label_to_string(label: str) -> str:
    return tokenize(label)


def name_to_string(uri: str) -> str:
    uri = (
        uri.replace("_", " ")
        .replace("-", " ")
        .replace(".", " ")
        .replace("/", " ")
        .replace('"', " ")
        .replace("'", " ")
    )
    uri = " ".join(camelcase.findall(uri))
    return tokenize(uri)
