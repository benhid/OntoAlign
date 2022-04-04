import argparse
import json
from pathlib import Path

from gensim.models import Word2Vec

from lib.Encoder import to_samples
from lib.SiameseNetwork import siamese_nn_predict

if __name__ == "__main__":
    # Initiate the parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "candidates",
        type=Path,
        default="logmap_output/logmap_overestimation.txt",
        help="Path to candidate mappings, e.g., LogMap overlapping mappings",
    )
    parser.add_argument(
        "--left-names",
        type=Path,
        default="left_names.json",
        help="Path to re-extracted class names of each class.",
    )
    parser.add_argument(
        "--right-names",
        type=Path,
        default="right_names.txt",
        help="Path to pre-extracted class names of each class.",
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
        default="word-con+vector",
        help="Mapping encoding type (any of: word-avg, word-con, word-avg+vector, word-con+vector).",
    )
    parser.add_argument(
        "--nn-dir",
        type=Path,
        default=Path("model"),
        help="Path for the inputs models.",
    )

    # Read arguments from the command line.
    args = parser.parse_args()

    # Reading files.
    with open(args.left_names, "r") as infile:
        left_names = json.load(infile)

    with open(args.right_names, "r") as infile:
        right_names = json.load(infile)

    left_tensors = {}
    if args.left_ltn:
        with open(args.left_ltn, "r") as infile:
            left_tensors = json.load(infile)

    right_tensors = {}
    if args.right_ltn:
        with open(args.right_ltn, "r") as infile:
            right_tensors = json.load(infile)

    # Reading LogMap candidates.
    with open(args.candidates, "r") as infile:
        candidates = infile.readlines()

    mappings = []

    for i, line in enumerate(candidates):
        mapping = line.strip().split("|")

        c1, c2 = mapping[0:2]

        n1 = left_names.get(c1)
        n2 = right_names.get(c2)

        if n1 and n2:
            for l1, l2 in [(x, y) for x in n1 for y in n2]:
                m = f"{i + 1}|{c1}|{c2}|{l1}|{l2}"
                mappings.append(m)

    print("%d candidates" % len(mappings))

    # Load OWL2Vec models.
    left_owl2vec_model = Word2Vec.load(str(args.left_owl2vec))
    right_owl2vec_model = Word2Vec.load(str(args.right_owl2vec))

    # Load Word2Vec models.
    left_wv_model = Word2Vec.load(str(args.left_w2v))
    right_wv_model = Word2Vec.load(str(args.right_w2v))

    X1, X2 = to_samples(
        mappings=mappings,
        left_owl2vec_model=left_owl2vec_model,
        right_owl2vec_model=right_owl2vec_model,
        left_wv_model=left_wv_model,
        right_wv_model=right_wv_model,
        left_tensors=left_tensors,
        right_tensors=right_tensors,
        encoder_type=args.encoder_type,
    )

    test_distances = siamese_nn_predict(test_x1=X1, test_x2=X2, nn_dir=args.nn_dir)
    test_scores = 1 - test_distances

    with open("prediction.txt", "w") as f:
        for i, mapping in enumerate(mappings):
            f.write("%s|%.3f\n" % (mapping, test_scores[i]))
