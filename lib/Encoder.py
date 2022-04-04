import numpy as np
from gensim.models.word2vec import Word2Vec


def encoder_tensors(v: str, g, dim) -> np.array:
    if v in g:
        return g[v]
    else:
        return np.zeros(dim)


def encoder_vector(v: str, wv_model: Word2Vec) -> np.array:
    wv_dim = wv_model.vector_size
    if v in wv_model.wv:
        return wv_model.wv[v]
    else:
        return np.zeros(wv_dim)


def encoder_path_con(
    path: list[str], wv_model: Word2Vec, label_num: int = 2, word_num: int = 3
) -> np.array:
    wv_dim = wv_model.vector_size
    path = (
        path[0:label_num]
        if len(path) >= label_num
        else path + ["NaN"] * (label_num - len(path))
    )
    sequence = []
    for label in path:
        words = label.split()
        words = (
            words[0:word_num]
            if len(words) >= word_num
            else words + ["NaN"] * (word_num - len(words))
        )
        sequence += words
    e = np.zeros((len(sequence), wv_dim))
    for i, word in enumerate(sequence):
        if word not in wv_model.wv:
            e[i, :] = np.zeros(wv_dim)
        else:
            e[i, :] = wv_model.wv[word]
    return e


def encoder_words_con(
    words: list[str], wv_model: Word2Vec, word_num: int = 3
) -> np.array:
    wv_dim = wv_model.vector_size
    words = (
        words[0:word_num]
        if len(words) >= word_num
        else words + ["NaN"] * (word_num - len(words))
    )
    e = np.zeros((word_num, wv_dim))
    for i, word in enumerate(words):
        if word not in wv_model.wv:
            e[i, :] = np.zeros(wv_dim)
        else:
            e[i, :] = wv_model.wv[word]
    return e


def encoder_words_avg(words: list[str], wv_model: Word2Vec) -> np.array:
    """
    Vector averaging means that the resulting vector is insensitive to the order of the words, i.e.,
    loses the word order in the same way as the standard bag-of-words models do.

    TODO - http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
    """
    wv_dim = wv_model.vector_size
    num, v = 0, np.zeros(wv_dim)
    for token in words:
        if token in wv_model.wv:
            num += 1
            v += wv_model.wv[token]
    avg = (v / num) if num > 0 else v
    return avg


def mapping_encoder(
    cls: str,
    label: str,
    owl2vec_model: Word2Vec,
    wv_model: Word2Vec,
    *,
    encoder_type: str,
):
    wv_dim = wv_model.vector_size
    words = label.split()
    if encoder_type == "vector":
        e = np.zeros((1, wv_dim))
        e[0, :] = encoder_vector(v=cls, wv_model=owl2vec_model)
    elif encoder_type == "word-avg":
        e = np.zeros((1, wv_dim))
        e[0, :] = encoder_words_avg(words=words, wv_model=wv_model)
    elif encoder_type == "word-con":
        # !!! el mejor
        # 2 tipo de eval: Threshold: 0.90, precision: 0.715, recall: 0.855, f1: 0.779
        e = np.zeros((3, wv_dim))
        e[0:, :] = encoder_words_con(words=words, wv_model=wv_model, word_num=3)
    elif encoder_type == "word-avg+vector":
        e = np.zeros((2, wv_dim))
        e[0, :] = encoder_vector(v=cls, wv_model=owl2vec_model)
        e[1, :] = encoder_words_avg(words=words, wv_model=wv_model)
    elif encoder_type == "word-con+vector":
        e = np.zeros((3, wv_dim))
        e[0, :] = encoder_vector(v=cls, wv_model=owl2vec_model)
        e[1:, :] = encoder_words_con(words=words, wv_model=wv_model, word_num=2)
        # print(encoder_type, cls, words, '\n', e[0, :5], e[1, :5], e[2, :5])
    elif encoder_type == "path-con+vector":
        e = np.zeros((2, wv_dim))
        e[0, :] = encoder_vector(v=cls, wv_model=owl2vec_model)
        e[1, :] = encoder_words_avg(words=[cls] + words, wv_model=wv_model)
    else:
        e = np.zeros((1, wv_dim))
    return e


def load_samples(
    mappings,
    left_owl2vec_model: Word2Vec,
    right_owl2vec_model: Word2Vec,
    left_wv_model: Word2Vec,
    right_wv_model: Word2Vec,
    left_tensors: dict,
    right_tensors: dict,
    *,
    encoder_type: str,
):
    assert left_owl2vec_model.vector_size == left_wv_model.vector_size, ""
    assert right_owl2vec_model.vector_size == right_wv_model.vector_size, ""

    left_wv_dim = left_wv_model.vector_size
    right_wv_dim = right_wv_model.vector_size

    num = len(mappings)

    if encoder_type == "vector":
        left_length, right_length = 1, 1
    elif encoder_type == "word-avg":
        left_length, right_length = 1, 1
    elif encoder_type == "word-con":
        left_length, right_length = 3, 3
    elif encoder_type == "word-avg+vector":
        left_length, right_length = 2, 2
    elif encoder_type == "word-con+vector":
        left_length, right_length = 3, 3
    elif encoder_type == "path-con+vector":
        left_length, right_length = 2, 2
    else:
        left_length, right_length = 1, 1

    # (height x weight x depth) or (batch_size x sequence_length x embedding_size)
    # see: https://jalammar.github.io/visual-numpy/
    #      https://www.w3resource.com/python-exercises/numpy/index-array.php
    X1 = np.zeros((num, left_length, left_wv_dim))
    X2 = np.zeros((num, right_length, right_wv_dim))
    Y = np.zeros((num, 2))

    print(f"X1 shape: ({X1.shape}), Y shape: {Y.shape}")

    for i in range(num):
        mapping = mappings[i].strip().split("|")

        c1, c2, l1, l2 = mapping[1:]

        X1[i] = mapping_encoder(
            cls=c1,
            label=l1,
            owl2vec_model=left_owl2vec_model,
            wv_model=left_wv_model,
            encoder_type=encoder_type,
        )
        X2[i] = mapping_encoder(
            cls=c2,
            label=l1,
            owl2vec_model=right_owl2vec_model,
            wv_model=right_wv_model,
            encoder_type=encoder_type,
        )
        Y[i] = (
            np.array([1.0, 0.0]) if mapping[0].startswith("-") else np.array([0.0, 1.0])
        )

    return X1, X2, Y, num


def to_samples(
    mappings,
    left_owl2vec_model: Word2Vec,
    right_owl2vec_model: Word2Vec,
    left_wv_model: Word2Vec,
    right_wv_model: Word2Vec,
    left_tensors: dict,
    right_tensors: dict,
    *,
    encoder_type: str,
):
    left_wv_dim = left_wv_model.vector_size
    right_wv_dim = right_wv_model.vector_size

    num = len(mappings)

    if encoder_type == "vector":
        left_length, right_length = 1, 1
    elif encoder_type == "word-avg":
        left_length, right_length = 1, 1
    elif encoder_type == "word-con":
        left_length, right_length = 3, 3
    elif encoder_type == "word-avg+vector":
        left_length, right_length = 2, 2
    elif encoder_type == "word-con+vector":
        left_length, right_length = 3, 3
    elif encoder_type == "path-con+vector":
        left_length, right_length = 2, 2
    else:
        left_length, right_length = 1, 1

    X1 = np.zeros((num, left_length, left_wv_dim))
    X2 = np.zeros((num, right_length, right_wv_dim))

    for i in range(num):
        mapping = mappings[i].strip().split("|")

        c1, c2, l1, l2 = mapping[1:]

        X1[i] = mapping_encoder(
            cls=c1,
            label=l1,
            owl2vec_model=left_owl2vec_model,
            wv_model=left_wv_model,
            encoder_type=encoder_type,
        )
        X2[i] = mapping_encoder(
            cls=c2,
            label=l2,
            owl2vec_model=right_owl2vec_model,
            wv_model=right_wv_model,
            encoder_type=encoder_type,
        )

    return X1, X2
