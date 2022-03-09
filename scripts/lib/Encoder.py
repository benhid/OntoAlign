import csv
from typing import List

import numpy as np
from gensim.models.word2vec import Word2Vec
from nltk.tokenize import word_tokenize

from .Label import parse_uri_name


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


def path_encoder_class_concat(path, class_num, wv_model) -> np.array:
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

    # `mappings` contains 3 lines per map (original + name + empty), thus the total number of mappings are:
    num = int(len(mappings) / 3)

    X1 = np.zeros((num, 3, left_wv_dim))  # one Set per mapping, 3 rows per set, `left_wv_dim` columns
    X2 = np.zeros((num, 3, right_wv_dim))
    Y = np.zeros((num, 2))

    for i in range(0, len(mappings), 3):
        class_mapping = mappings[i].split("|")
        left_c, right_c = class_mapping[2], class_mapping[3]

        name_mapping = mappings[i + 1].split("|")
        p1 = [x for x in list(csv.reader([name_mapping[2]], delimiter=",", quotechar='"'))[0]]
        p2 = [x for x in list(csv.reader([name_mapping[3]], delimiter=",", quotechar='"'))[0]]

        # Path type is 'uri+label', so we want to construct the class path using the URI name and labels.
        p1 = [left_c.split(":")[1]] + p1
        p2 = [right_c.split(":")[1]] + p2

        j = int(i / 3)

        # Embeds a path by concatenating the embeddings of its classes.
        X1[j] = path_encoder_class_concat(
            path=p1,
            wv_model=left_wv_model,
            class_num=3,
        )
        X2[j] = path_encoder_class_concat(
            path=p2,
            wv_model=right_wv_model,
            class_num=3,
        )
        Y[j] = (
            np.array([1.0, 0.0]) if name_mapping[0].startswith("neg") else np.array([0.0, 1.0])
        )

    return X1, X2, Y, num


def to_samples(mappings, mappings_names, left_wv_model: Word2Vec, right_wv_model: Word2Vec):
    # TODO - Can we use `load_samples()`?
    left_wv_dim = left_wv_model.vector_size
    right_wv_dim = right_wv_model.vector_size

    num = len(mappings)

    X1 = np.zeros((num, 3, left_wv_dim))
    X2 = np.zeros((num, 3, right_wv_dim))

    for i in range(num):

        class_mapping = mappings[i].split("|")
        left_c, right_c = class_mapping[2], class_mapping[3]

        name_mapping = mappings_names[i].split("|")
        p1 = [x for x in list(csv.reader([name_mapping[2]], delimiter=",", quotechar='"'))[0]]
        p2 = [x for x in list(csv.reader([name_mapping[3]], delimiter=",", quotechar='"'))[0]]

        # Path type is 'uri+label', so we want to construct the class path using the URI name and labels.
        p1 = [left_c.split(":")[1]] + p1
        p2 = [right_c.split(":")[1]] + p2

        # Embeds a path by concatenating the embeddings of its classes.
        X1[i] = path_encoder_class_concat(
            path=p1,
            wv_model=left_wv_model,
            class_num=3,
        )
        X2[i] = path_encoder_class_concat(
            path=p2,
            wv_model=right_wv_model,
            class_num=3,
        )

    return X1, X2
