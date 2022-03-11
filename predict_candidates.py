import argparse
import json
from pathlib import Path

from gensim.models import Word2Vec

from lib.Encoder import to_samples
from lib.Label import name_to_string
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
    "--left_w2v",
    type=Path,
    default="w2v/",
    help="Path to OWL2Vec or Word2Vec model of the left ontology",
)
parser.add_argument(
    "--right_w2v",
    type=Path,
    default="w2v/",
    help="Path to OWL2Vec or Word2Vec model of the right ontology",
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
    help="Path for the inputs models.",
)


if __name__ == "__main__":
    # Read arguments from the command line.
    args = parser.parse_args()

    # Reading files.
    with open(args.left_names, "r") as infile:
        left_names = json.load(infile)

    with open(args.right_names, "r") as infile:
        right_names = json.load(infile)

    # Reading LogMap candidates.
    with open(args.candidates, "r") as infile:
        candidates = infile.readlines()

    mappings, mappings_n = [], []

    for i, line in enumerate(candidates):
        mapping = line.strip().split("|")

        c1 = mapping[0]
        c2 = mapping[1]

        n1 = left_names.get(c1)
        n2 = right_names.get(c2)

        # TODO - ¿aquí como lo hago?

        if n1 and n2:
            for l1, l2 in [(x, y) for x in n1 for y in n2]:
                origin = "%d|%s|%s" % (i + 1, c1, c2)
                label = "%d|%s|%s" % (i + 1, l1, l2)
                mappings.append(origin)
                mappings_n.append(label)
        """
        if n1 and n2:
            origin = "%d|%s|%s" % (i + 1, c1, c2)
            label = "%d|%s|%s" % (i + 1, n1[0], n2[0])
            mappings.append(origin)
            mappings_n.append(label)
        """

    print("%d mappings in total" % len(mappings))

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
