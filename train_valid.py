import argparse
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec

from lib.Encoder import load_samples
from lib.Evaluator import threshold_searching
from lib.SiameseNetwork import siamese_nn_predict, siamese_nn_train

# Initiate the parser.
parser = argparse.ArgumentParser()
parser.add_argument("--train_mappings", type=Path, default="train_mappings.txt")
parser.add_argument("--valid_mappings", type=Path, default="validation_mappings.txt")
parser.add_argument(
    "--left_w2v",
    type=Path,
    help="Path to OWL2Vec or Word2Vec model of the left ontology",
)
parser.add_argument(
    "--right_w2v",
    type=Path,
    help="Path to OWL2Vec or Word2Vec model of the right ontology",
)
parser.add_argument(
    "--nn_dir",
    type=Path,
    default=Path("model"),
    help="Path for the output models.",
)

if __name__ == "__main__":
    # Read arguments from the command line.
    args = parser.parse_args()

    with open(args.train_mappings, "r") as infile:
        train_mappings = infile.readlines()

    with open(args.valid_mappings, "r") as infile:
        valid_mappings = infile.readlines()

    left_wv_model = Word2Vec.load(str(args.left_w2v))
    right_wv_model = Word2Vec.load(str(args.right_w2v))

    train_X1, train_X2, train_Y, train_num = load_samples(
        mappings=train_mappings,
        left_wv_model=left_wv_model,
        right_wv_model=right_wv_model,
    )

    shuffle_indices = np.random.permutation(np.arange(train_num))
    train_X1, train_X2, train_Y = (
        train_X1[shuffle_indices],
        train_X2[shuffle_indices],
        train_Y[shuffle_indices],
    )

    valid_X1, valid_X2, valid_Y, valid_num = load_samples(
        mappings=valid_mappings,
        left_wv_model=left_wv_model,
        right_wv_model=right_wv_model,
    )

    def train(X1, X2, Y, nn_dir):
        siamese_nn_train(train_x1=X1, train_x2=X2, y_train=Y, nn_dir=nn_dir)

    def valid(X1, X2, Y, nn_dir):
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

    train(X1=train_X1, X2=train_X2, Y=train_Y, nn_dir=args.nn_dir)

    threshold, f1, p, r, acc = valid(
        X1=valid_X1, X2=valid_X2, Y=valid_Y, nn_dir=args.nn_dir
    )

    print(
        "Threshold: %.2f, precision: %.3f, recall: %.3f, f1: %.3f, acc: %.3f"
        % (threshold, p, r, f1, acc)
    )
