import spacy
import re

nlp = spacy.load("en_core_web_sm")

camelcase = re.compile(r'.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')


def tokenize(item: str):
    doc = nlp(str(item), disable=['tagger', 'parser', 'ner', 'lemmatizer', 'textcat'])
    return ' '.join(token.text.lower() for token in doc if not token.is_punct)


def label_to_string(label: str) -> str:
    name_str = tokenize(label)
    return f'"{name_str}"'


def name_to_string(uri: str) -> str:
    uri = (
        uri.replace("_", " ")
        .replace("-", " ")
        .replace(".", " ")
        .replace("/", " ")
        .replace('"', " ")
        .replace("'", " ")
    )
    uri = ' '.join(camelcase.findall(uri))
    name_str = tokenize(uri)
    return f'"{name_str}"'
