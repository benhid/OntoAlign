import csv
from typing import List

import numpy as np
from gensim.models.word2vec import Word2Vec
from nltk.tokenize import word_tokenize

from scripts.lib.Label import parse_uri_name


def to_words(item: str) -> List[str]:
    if item.startswith("http://"):
        if "#" in item:
            uri_name = item.split("#")[1]
        else:
            uri_name = item.split("/")[-1]
        words_str = parse_uri_name(uri_name)
        words = words_str.split(" ")
    else:
        item = (
            item.replace("_", " ")
            .replace("-", " ")
            .replace(".", " ")
            .replace("/", " ")
            .replace('"', " ")
            .replace("'", " ")
            .replace("\\", " ")
            .replace("(", " ")
            .replace(")", " ")
        )
        tokenized_line = " ".join(word_tokenize(item))
        words = [word for word in tokenized_line.lower().split()]
    return words


def path_encoder_word_avg(name_path, wv_model):
    wv_dim = wv_model.vector_size
    num, v = 0, np.zeros(wv_dim)
    for item in name_path:
        for word in to_words(item=item):
            if word in wv_model.wv:
                num += 1
                v += wv_model.wv[word]
    avg = (v / num) if num > 0 else v
    return avg


def path_encoder_class_con(path, class_num, wv_model) -> np.array:
    wv_dim = wv_model.vector_size
    path = (
        path[0:class_num]
        if len(path) >= class_num
        else path + ["NaN"] * (class_num - len(path))
    )
    e = np.zeros((len(path), wv_dim))
    for i, item in enumerate(path):
        if item == "NaN":
            e[i, :] = np.zeros(wv_dim)
        else:
            e[i, :] = path_encoder_word_avg(name_path=[item], wv_model=wv_model)
    return e


def load_samples(mappings, left_wv_model: Word2Vec, right_wv_model: Word2Vec):
    left_wv_dim = left_wv_model.vector_size
    right_wv_dim = right_wv_model.vector_size

    num = int(len(mappings) / 3)

    # Path type is set to 'uri+label' (the uri name and label of the class).
    X1 = np.zeros((num, 3, left_wv_dim))
    X2 = np.zeros((num, 3, right_wv_dim))
    Y = np.zeros((num, 2))

    for i in range(0, len(mappings), 3):
        name_mapping = mappings[i + 1]
        tmp = name_mapping.split("|")

        p1 = [x for x in list(csv.reader([tmp[2]], delimiter=",", quotechar='"'))[0]]
        p2 = [x for x in list(csv.reader([tmp[3]], delimiter=",", quotechar='"'))[0]]

        mapping = mappings[i].strip().split("|")
        left_c, right_c = mapping[2], mapping[3]

        p1 = [left_c.split(":")[1]] + p1
        p2 = [right_c.split(":")[1]] + p2

        j = int(i / 3)

        # Embeds a path by concatenating the embeddings of its classes.
        X1[j] = path_encoder_class_con(
            path=p1,
            wv_model=left_wv_model,
            class_num=3,
        )
        X2[j] = path_encoder_class_con(
            path=p2,
            wv_model=right_wv_model,
            class_num=3,
        )
        Y[j] = (
            np.array([1.0, 0.0]) if tmp[0].startswith("neg") else np.array([0.0, 1.0])
        )

    return X1, X2, Y, num


def to_samples(mappings, mappings_n, left_wv_model: Word2Vec, right_wv_model: Word2Vec):
    left_wv_dim = left_wv_model.vector_size
    right_wv_dim = right_wv_model.vector_size

    num = len(mappings_n)

    X1 = np.zeros((num, 3, left_wv_dim))
    X2 = np.zeros((num, 3, right_wv_dim))

    for i in range(num):
        tmp = mappings_n[i].split("|")
        p1 = [x for x in list(csv.reader([tmp[0]], delimiter=",", quotechar='"'))[0]]
        p2 = [x for x in list(csv.reader([tmp[1]], delimiter=",", quotechar='"'))[0]]

        tmp = mappings[i].split("|")
        left_c, right_c = tmp[1], tmp[2]

        p1 = [left_c.split(":")[1]] + p1
        p2 = [right_c.split(":")[1]] + p2

        # Embeds a path by concatenating the embeddings of its classes.
        X1[i] = path_encoder_class_con(
            path=p1,
            wv_model=left_wv_model,
            class_num=3,
        )
        X2[i] = path_encoder_class_con(
            path=p2,
            wv_model=right_wv_model,
            class_num=3,
        )

    return X1, X2
