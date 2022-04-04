import numpy as np

from lib.Encoder import encoder_word_avg, path_encoder_class_concat


class _Word2Vec:
    def __init__(self, vector_size: int = 5):
        self.wv = {
            "a": [1] * vector_size,
            "b": [0.5] * vector_size,
            "c": [0.75] * vector_size,
        }
        self.vector_size = vector_size


def test_path_encoder_word_avg():
    name_path = ["a", "b", "c"]
    wv_model = _Word2Vec(vector_size=5)
    encoded = encoder_word_avg(name_path=name_path, wv_model=wv_model)
    assert np.array_equal(encoded, np.array([0.75] * wv_model.vector_size))


def test_path_encoder_class_concat():
    path = ["a", "b", "c", "d"]
    class_num = 2
    wv_model = _Word2Vec(vector_size=5)
    encoded = path_encoder_class_concat(
        path=path, class_num=class_num, wv_model=wv_model
    )
    assert np.array_equal(
        encoded, np.array([[1] * wv_model.vector_size, [0.5] * wv_model.vector_size])
    )


def test_path_encoder_class_concat_fill():
    path = ["a"]
    class_num = 2
    wv_model = _Word2Vec(vector_size=5)
    encoded = path_encoder_class_concat(
        path=path, class_num=class_num, wv_model=wv_model
    )
    assert np.array_equal(
        encoded, np.array([[1] * wv_model.vector_size, [0] * wv_model.vector_size])
    )
