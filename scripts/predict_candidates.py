import argparse
import json
from datetime import datetime
from pathlib import Path

from gensim.models import Word2Vec
from lib.Encoder import to_samples
from lib.Label import get_label, replace_with_prefix
from lib.SiameseNetwork import siamese_nn_predict

# Initiate the parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    "candidates",
    type=Path,
    default="logmap_output/logmap_overestimation.txt",
    help="Path to candidate mappings, e.g., LogMap overlapping mappings",
)
parser.add_argument(
    "--left_paths", type=Path, help="Path to pre-extracted paths of each class."
)
parser.add_argument(
    "--right_paths", type=Path, help="Path to pre-extracted paths of each class."
)
parser.add_argument(
    "--left_names",
    type=Path,
    help="Path to re-extracted class names of each class.",
)
parser.add_argument(
    "--right_names",
    type=Path,
    help="Path to pre-extracted class names of each class.",
)
parser.add_argument(
    "--left_w2v_dir",
    type=Path,
    default="w2v/",
    help="OWL2Vec or Word2Vec of the left ontology",
)
parser.add_argument(
    "--right_w2v_dir",
    type=Path,
    default="w2v/",
    help="OWL2Vec or Word2Vec of the right ontology",
)
parser.add_argument(
    "--keep_uri",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Set it to keep URI in the sample instead of using the path labels.",
)
parser.add_argument(
    "--nn_dir",
    type=Path,
    default=Path("model"),
    help="Path for the output models.",
)

# Read arguments from the command line.
args = parser.parse_args()


if __name__ == "__main__":
    # Reading files.
    with open(args.left_names, "r") as infile:
        left_names = json.load(infile)

    with open(args.right_names, "r") as infile:
        right_names = json.load(infile)

    with open(args.left_paths, "r") as infile:
        left_paths = [line.strip().split(",") for line in infile.readlines()]

    with open(args.right_paths, "r") as infile:
        right_paths = [line.strip().split(",") for line in infile.readlines()]

    # Reading LogMap candidates.
    with open(args.candidates, "r") as infile:
        candidates = infile.readlines()

    # TODO - Comment.
    mappings, mappings_n = [], []

    for i, line in enumerate(candidates):
        m = line.strip().split(", ")[1] if ", " in line else line.strip()
        m_split = m.split("|")

        c1 = replace_with_prefix(uri=m_split[0])
        c2 = replace_with_prefix(uri=m_split[1])

        l1 = get_label(
            cls=c1,
            paths=left_paths,
            names=left_names,
            keep_uri=args.keep_uri,
        )
        l2 = get_label(
            cls=c2,
            paths=left_paths,
            names=left_names,
            keep_uri=args.keep_uri,
        )

        origin = "i=%d|%s|%s" % (i + 1, c1, c2)
        name = "%s|%s" % (l1, l2)

        mappings.append(origin)
        mappings_n.append(name)

    left_wv_model = Word2Vec.load(str(args.left_w2v))
    right_wv_model = Word2Vec.load(str(args.right_w2v))

    X1, X2 = to_samples(
        mappings=mappings,
        mappings_n=mappings_n,
        left_wv_model=left_wv_model,
        right_wv_model=right_wv_model,
    )

    test_distances = siamese_nn_predict(test_x1=X1, test_x2=X2, nn_dir=args.nn_dir)
    test_scores = 1 - test_distances

    with open("prediction.txt", "w") as f:
        for i, mapping in enumerate(mappings):
            f.write("%s|%.3f\n" % (mapping, test_scores[i]))
            f.write("%s\n" % mappings_n[i])
            f.write("\n")
