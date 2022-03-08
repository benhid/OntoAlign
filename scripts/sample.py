import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from lib.Label import get_label, replace_with_prefix

# Initiate the parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    "anchors",
    type=Path,
    default="logmap_output/logmap_anchors.txt",
    help="Path to the LogMap anchors file.",
)
parser.add_argument(
    "--train_rate",
    type=float,
    default=1.0,
    help="Can be set to 1.0 (to use all the seeds as the training set and 20% of them as the validation set)"
    "or a float smaller than 1.0 (where train_rate of all the samples are used as the training set and the remaining are used as the validation set).",
)
parser.add_argument("--sample_duplicate", type=int, default=2)
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
    "--keep_uri",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Set it to keep URI in the sample instead of using the path labels.",
)
parser.add_argument(
    "--anchor_branch_conflict",
    default=True,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--generate_negative_sample",
    default=True,
    action=argparse.BooleanOptionalAction,
)

# Read arguments from the command line.
args = parser.parse_args()

# Class disjointness constraints for HeLis and FoodOn.
branch_conflicts = [
    [
        '","nutrient"',
        '"food product type","material entity","independent continuant","continuant","entity"',
    ],
    [
        '"basic food","food"',
        '"food source","environmental material","fiat object part","material entity","independent continuant","continuant","entity"',
    ],
    ['"basic food","food"', '"organism","material entity"'],
    ['"basic food","food"', '"chemical entity","material entity"'],
]


def violates_branch_conflict(l1: str, l2: str) -> bool:
    for conflict in branch_conflicts:
        if conflict[0] in l1 and conflict[1] in l2:
            return True
    return False


def negative_sampling(
    mappings: List[List[str]],
    left_paths: List[List[str]],
    right_paths: List[List[str]],
    left_names: Dict[str, List[str]],
    right_names: Dict[str, List[str]],
) -> List[List[str]]:
    """
    For each mapping (c1,c2), we generate one negative sample replacing c1 with a class randomly selected
    from the left ontology, and we generate a second negative sample replacing c2 with a class randomly selected
    from the right ontology.

    TODO - donde "Note that the random replacements could produce positive samples from Ms; we discard any such negative samples."?
    """
    neg_mappings = []

    for mapping in mappings:
        line = mapping[0].split("|")
        i, c1, c2 = line[0], line[2], line[3]

        line = mapping[1].split("|")
        l1, l2 = line[2], line[3]

        neg_c1 = random.sample(list(left_classes - {c1}), 1)[0]
        neg_l1 = get_label(
            cls=neg_c1,
            paths=left_paths,
            names=left_names,
            keep_uri=args.keep_uri,
        )

        if not neg_l1 == '""' and not l2 == '""':
            origin = "neg-%s-h|origin|%s|%s" % (i, neg_c1, c2)
            name = "neg-%s-h|name|%s|%s" % (i, neg_l1, l2)
            neg_mappings.append([origin, name])

        neg_c2 = random.sample(list(right_classes - {c2}), 1)[0]
        neg_l2 = get_label(
            cls=neg_c2,
            paths=right_paths,
            names=right_names,
            keep_uri=args.keep_uri,
        )

        if not l1 == '""' and not neg_l2 == '""':
            origin = "neg-%s-f|origin|%s|%s" % (i, c1, neg_c2)
            name = "neg-%s-f|name|%s|%s" % (i, l1, neg_l2)
            neg_mappings.append([origin, name])

    return neg_mappings


if __name__ == "__main__":
    # Reading files.
    with open(args.left_names, "r") as infile:
        left_names = json.load(infile)

    with open(args.right_names, "r") as infile:
        right_names = json.load(infile)

    left_classes = set(left_names.keys())
    right_classes = set(right_names.keys())

    with open(args.left_paths, "r") as infile:
        left_paths = [line.strip().split(",") for line in infile.readlines()]

    with open(args.right_paths, "r") as infile:
        right_paths = [line.strip().split(",") for line in infile.readlines()]

    # Reading LogMap anchors.
    with open(args.anchors, "r") as infile:
        anchors = infile.readlines()

    # TODO - Add comment.
    mappings = []
    rule_violated_mappings = []

    for i, line in enumerate(anchors):
        line = line.strip().split("|")

        c1 = replace_with_prefix(line[0])
        c2 = replace_with_prefix(line[1])

        l1 = get_label(
            cls=c1,
            paths=left_paths,
            names=left_names,
            keep_uri=args.keep_uri,
        )
        l2 = get_label(
            cls=c2,
            paths=right_paths,
            names=right_names,
            keep_uri=args.keep_uri,
        )

        if not l1 == '""' and not l2 == '""':
            if args.anchor_branch_conflict and violates_branch_conflict(l1, l2):
                origin = "neg-%d|origin|%s|%s" % (i + 1, c1, c2)
                name = "neg-%d|name|%s|%s" % (i + 1, l1, l2)  # TODO name = label?
                rule_violated_mappings.append([origin, name])
            else:
                origin = "%d|origin|%s|%s" % (i + 1, c1, c2)
                name = "%d|name|%s|%s" % (i + 1, l1, l2)  # TODO name = label?
                mappings.append([origin, name])

    print(
        "%d mappings in total, %d mappings violate the rules"
        % (len(mappings), len(rule_violated_mappings))
    )

    random.shuffle(mappings)

    ratio = round(len(mappings) * args.train_rate)

    train_mappings = mappings[0:ratio]
    if args.train_rate < 1.0:
        validation_mappings = mappings[ratio:]
    else:
        validation_mappings = mappings[round(len(mappings) * 0.8) :]

    # We also adopt anchor mappings that violate the class disjointness constraints as negative samples and
    # randomly partition them into a training set and a validation set with the same ratio:

    random.shuffle(rule_violated_mappings)

    rv_ratio = round(len(rule_violated_mappings) * args.train_rate)

    train_rv_mappings = rule_violated_mappings[0:rv_ratio]
    if args.train_rate < 1.0:
        valid_rv_mappings = rule_violated_mappings[rv_ratio:]
    else:
        valid_rv_mappings = rule_violated_mappings[
            round(len(rule_violated_mappings) * 0.8) :
        ]

    if args.generate_negative_sample:
        train_mappings = (
            train_mappings * args.sample_duplicate
            + train_rv_mappings * args.sample_duplicate
            + negative_sampling(
                mappings=train_mappings,
                left_paths=left_paths,
                right_paths=right_paths,
                left_names=left_names,
                right_names=right_names,
            )
        )
        validation_mappings = (
            validation_mappings * args.sample_duplicate
            + valid_rv_mappings * args.sample_duplicate
            + negative_sampling(
                mappings=validation_mappings,
                left_paths=left_paths,
                right_paths=right_paths,
                left_names=left_names,
                right_names=right_names,
            )
        )
    else:
        train_mappings = train_mappings + train_rv_mappings
        validation_mappings = validation_mappings + valid_rv_mappings

    with open("train_mappings.txt", "w") as f2:
        for m in train_mappings:
            f2.write(m[0] + "\n")
            f2.write(m[1] + "\n")
            f2.write("\n")

    with open("validation_mappings.txt", "w") as f2:
        for m in validation_mappings:
            f2.write(m[0] + "\n")
            f2.write(m[1] + "\n")
            f2.write("\n")

    print("-- Done --")
