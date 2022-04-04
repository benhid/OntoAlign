import argparse
import json
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec

from lib.Encoder import load_samples
from lib.Evaluator import threshold_searching
from lib.SiameseNetwork import siamese_nn_predict, siamese_nn_train


def train(X1, X2, Y, nn_dir):
    siamese_nn_train(train_x1=X1, train_x2=X2, y_train=Y, nn_dir=nn_dir)


def valid(X1, X2, Y, nn_dir, valid_num):
    valid_distances = siamese_nn_predict(test_x1=X1, test_x2=X2, nn_dir=nn_dir)
    valid_scores = 1 - valid_distances
    (
        max_alpha,
        max_valid_f1,
        max_valid_p,
        max_valid_r,
        max_valid_acc,
    ) = threshold_searching(Y=Y[:, 1], scores=valid_scores, num=valid_num)
    return max_alpha, max_valid_f1, max_valid_p, max_valid_r, max_valid_acc


if __name__ == "__main__":
    # Initiate the parser.
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-mappings", type=Path, default="train_mappings.txt")
    parser.add_argument(
        "--valid-mappings", type=Path, default="validation_mappings.txt"
    )
    parser.add_argument(
        "--left-owl2vec",
        type=Path,
        help="Path to OWL2Vec model of the left ontology",
    )
    parser.add_argument(
        "--right-owl2vec",
        type=Path,
        help="Path to OWL2Vec model of the right ontology",
    )
    parser.add_argument(
        "--left-w2v",
        type=Path,
        help="Path to OWL2Vec or Word2Vec model of the left ontology",
    )
    parser.add_argument(
        "--right-w2v",
        type=Path,
        help="Path to OWL2Vec or Word2Vec model of the right ontology",
    )
    parser.add_argument(
        "--left-ltn",
        type=Path,
        default=None,
        help="Path to LTN tensors of the left ontology",
    )
    parser.add_argument(
        "--right-ltn",
        type=Path,
        default=None,
        help="Path to LTN tensors of the right ontology",
    )
    parser.add_argument(
        "--encoder-type",
        type=str,
        default=None,
        help="Mapping encoding type",
    )
    parser.add_argument(
        "--nn-dir",
        type=Path,
        default=Path("model"),
        help="Path for the output models.",
    )

    # Read arguments from the command line.
    args = parser.parse_args()

    left_tensors = {}
    if args.left_ltn:
        with open(args.left_ltn, "r") as infile:
            left_tensors = json.load(infile)

    right_tensors = {}
    if args.right_ltn:
        with open(args.right_ltn, "r") as infile:
            right_tensors = json.load(infile)

    # Read mappings.
    with open(args.train_mappings, "r") as infile:
        train_mappings = infile.readlines()

    with open(args.valid_mappings, "r") as infile:
        valid_mappings = infile.readlines()

    # Load OWL2Vec models.
    left_owl2vec_model = Word2Vec.load(str(args.left_owl2vec))
    right_owl2vec_model = Word2Vec.load(str(args.right_owl2vec))

    # Load Word2Vec models.
    left_wv_model = Word2Vec.load(str(args.left_w2v))
    right_wv_model = Word2Vec.load(str(args.right_w2v))

    all_encoder_types = [
        "vector",
        "word-avg",
        "word-con",
        "word-avg+vector",
        "word-con+vector",
        "path-con+vector",
        "",
    ]

    encoder_types = [args.encoder_type] if args.encoder_type else all_encoder_types

    for encoder_type in encoder_types:
        print(f"--- START encoder_type:{encoder_type} ---")

        train_X1, train_X2, train_Y, train_num = load_samples(
            mappings=train_mappings,
            left_owl2vec_model=left_owl2vec_model,
            right_owl2vec_model=right_owl2vec_model,
            left_wv_model=left_wv_model,
            right_wv_model=right_wv_model,
            left_tensors=left_tensors,
            right_tensors=right_tensors,
            encoder_type=encoder_type,
        )

        shuffle_indices = np.random.permutation(np.arange(train_num))
        train_X1, train_X2, train_Y = (
            train_X1[shuffle_indices],
            train_X2[shuffle_indices],
            train_Y[shuffle_indices],
        )

        train(X1=train_X1, X2=train_X2, Y=train_Y, nn_dir=args.nn_dir)

        valid_X1, valid_X2, valid_Y, valid_num = load_samples(
            mappings=valid_mappings,
            left_owl2vec_model=left_owl2vec_model,
            right_owl2vec_model=right_owl2vec_model,
            left_wv_model=left_wv_model,
            right_wv_model=right_wv_model,
            left_tensors=left_tensors,
            right_tensors=right_tensors,
            encoder_type=encoder_type,
        )

        threshold, f1, p, r, acc = valid(
            X1=valid_X1,
            X2=valid_X2,
            Y=valid_Y,
            nn_dir=args.nn_dir,
            valid_num=valid_num,
        )

        print(
            f"Best setting: Threshold: {threshold}, precision: {p}, recall: {r}, f1: {f1}, acc: {acc}\n"
        )
