import numpy as np
from gensim.models.word2vec import Word2Vec

from lib.Label import tokenize


def vector(item: str, wv_model: Word2Vec) -> np.array:
    if item in wv_model.wv:
        return wv_model.wv[item]
    else:
        return np.zeros(wv_model.vector_size)


def encoder_word_avg(item: str, wv_model: Word2Vec) -> np.array:
    """
    Vector averaging means that the resulting vector is insensitive to the order of the words.
    """
    wv_dim = wv_model.vector_size
    num, v = 0, np.zeros(wv_dim)
    for token in tokenize(item).split():
        if token in wv_model.wv:
            num += 1
            v += wv_model.wv[token]
    avg = (v / num) if num > 0 else v
    return avg


def encoder_avg(item: str, uri: str, wv_model: Word2Vec, vec_type: str = 'word'):
    if vec_type == 'word':
        return encoder_word_avg(item=item, wv_model=wv_model)
    elif vec_type == 'uri':
        return vector(item=item, wv_model=wv_model)
    elif vec_type == 'uri+label':
        word_avg = encoder_word_avg(item=item, wv_model=wv_model)
        uri_avg = vector(item=uri, wv_model=wv_model)
        return np.concatenate((word_avg, uri_avg))
    else:
        raise AttributeError


def load_samples(mappings, left_wv_model: Word2Vec, right_wv_model: Word2Vec, vec_type: str = 'uri+label'):
    left_wv_dim = left_wv_model.vector_size
    right_wv_dim = right_wv_model.vector_size

    if vec_type == "uri+label":
        left_wv_dim *= 2
        right_wv_dim *= 2

    num = int(len(mappings) / 2)

    # (height x weight x depth) or (batch_size x sequence_length x embedding_size)
    # see: https://jalammar.github.io/visual-numpy/
    #      https://www.w3resource.com/python-exercises/numpy/index-array.php
    X1 = np.zeros((num, 1, left_wv_dim))
    X2 = np.zeros((num, 1, right_wv_dim))
    Y = np.zeros((num, 2))

    for i in range(0, len(mappings), 2):
        class_mapping = mappings[i].split("|")
        c1, c2 = class_mapping[1], class_mapping[2]

        name_mapping = mappings[i + 1].split("|")

        n1, n2 = name_mapping[1], name_mapping[2]

        j = int(i / 2)

        X1[j] = encoder_avg(
            item=n1,
            uri=c1,
            wv_model=left_wv_model,
            vec_type=vec_type
        )
        X2[j] = encoder_avg(
            item=n2,
            uri=c2,
            wv_model=right_wv_model,
            vec_type=vec_type
        )
        Y[j] = (
            np.array([1.0, 0.0])
            if name_mapping[0].startswith("-")
            else np.array([0.0, 1.0])
        )

    return X1, X2, Y, num

def to_samples(
    mappings, mappings_n, left_wv_model: Word2Vec, right_wv_model: Word2Vec, vec_type: str = 'uri+label'
):
    left_wv_dim = left_wv_model.vector_size
    right_wv_dim = right_wv_model.vector_size

    if vec_type == "uri+label":
        left_wv_dim *= 2
        right_wv_dim *= 2

    num = len(mappings)

    X1 = np.zeros((num, 1, left_wv_dim))
    X2 = np.zeros((num, 1, right_wv_dim))

    for i in range(len(mappings)):
        class_mapping = mappings[i].split("|")
        c1, c2 = class_mapping[1], class_mapping[2]

        name_mapping = mappings_n[i].split("|")

        n1, n2 = name_mapping[1], name_mapping[2]

        X1[i] = encoder_avg(
            item=n1,
            uri=c1,
            wv_model=right_wv_model,
            vec_type=vec_type
        )
        X2[i] = encoder_avg(
            item=n2,
            uri=c2,
            wv_model=right_wv_model,
            vec_type=vec_type
        )

    return X1, X2
