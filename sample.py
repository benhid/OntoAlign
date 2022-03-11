import argparse
import json
import random
from pathlib import Path

# Initiate the parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    "anchors",
    type=Path,
    default="logmap_output/logmap_anchors.txt",
    help="Path to LogMap anchors file.",
)
parser.add_argument(
    "--conflicting_mappings",
    type=Path,
    default="logmap_output/logmap_discarded_mappings.txt",
    help="Path to  LogMap discarded mappings file.",
)
parser.add_argument("--sample_duplicate", type=int, default=2)
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
    "--train_rate",
    type=float,
    default=1.0,
    help="Can be set to 1.0 (to use all the seeds as the training set and 20% of them as the validation set)"
    "or a float smaller than 1.0 (where train_rate of all the samples are used as the training set and the remaining are used as the validation set).",
)
parser.add_argument(
    "--anchor_branch_conflict",
    default=False,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--generate_negative_sample",
    default=True,
    action=argparse.BooleanOptionalAction,
)

# Class disjointness constraints.
branch_conflicts = [
    ['','']
]


def violates_branch_conflict(l1: str, l2: str) -> bool:
    for conflict in branch_conflicts:
        if conflict[0] in l1 and conflict[1] in l2:
            return True
    return False


def split_train_valid(mappings, train_rate):
    random.shuffle(mappings)
    ratio = round(len(mappings) * train_rate)
    train = mappings[0:ratio]
    if train_rate < 1.0:
        valid = mappings[ratio:]
    else:
        valid = mappings[round(len(mappings) * 0.8):]
    return train, valid


def negative_sampling(pos_mappings, left_names, right_names):
    neg_mappings = list()
    for line in pos_mappings:
        class_mapping = line[0].strip().split("|")

        idx = class_mapping[0]
        c1 = class_mapping[1]
        c2 = class_mapping[2]

        n1 = left_names.get(c1)
        n2 = right_names.get(c2)

        neg_c2 = random.sample(list(right_names.keys() - {c2}), 1)[0]
        neg_n2 = right_names.get(neg_c2)

        if n1 and neg_n2:
            origin = '-%s-f|%s|%s' % (idx, c1, neg_c2)
            name = '-%s-f|%s|%s' % (idx, n1, neg_n2)
            neg_mappings.append([origin, name])

        neg_c1 = random.sample(list(left_names.keys() - {c1}), 1)[0]
        neg_n1 = left_names.get(neg_c1)

        if neg_n1 and n2:
            origin = '-%s-h|%s|%s' % (idx, neg_c1, c2)
            name = '-%s-h|%s|%s' % (idx, neg_n1, n2)
            neg_mappings.append([origin, name])

    return neg_mappings


if __name__ == "__main__":
    # Read arguments from the command line.
    args = parser.parse_args()

    # Read files.
    with open(args.left_names, "r") as infile:
        left_names = json.load(infile)

    with open(args.right_names, "r") as infile:
        right_names = json.load(infile)

    # Read initial set of candidate mappings (anchors).
    with open(args.anchors, "r") as infile:
        anchors = infile.readlines()

    # Read conflict mappings from LogMap.
    with open(args.conflicting_mappings, "r") as infile:
        conflict_mappings = infile.readlines()

    mappings = []
    rule_violated_mappings = []

    for i, line in enumerate(anchors):
        mapping = line.strip().split("|")

        c1 = mapping[0]
        c2 = mapping[1]

        n1 = left_names[c1]
        n2 = right_names[c2]
        
        for l1, l2 in [(x, y) for x in n1 for y in n2]:
            if args.anchor_branch_conflict and violates_branch_conflict(l1, l2):
                origin = "-%d|%s|%s" % (i + 1, c1, c2)
                label = "-%d|%s|%s" % (i + 1, l1, l2)
                rule_violated_mappings.append([origin, label])
            else:
                origin = "%d|%s|%s" % (i + 1, c1, c2)
                label = "%d|%s|%s" % (i + 1, l1, l2)
                mappings.append([origin, label])

    print(
        "%d mappings in total, %d mappings violate the rules"
        % (len(mappings), len(rule_violated_mappings))
    )

    train_mappings, validation_mappings = split_train_valid(mappings, train_rate=args.train_rate)

    # We also adopt anchor mappings that violate the class disjointness constraints as negative samples and
    # randomly partition them into a training set and a validation set with the same ratio:
    train_rv_mappings, valid_rv_mappings = split_train_valid(rule_violated_mappings, train_rate=args.train_rate)

    if args.generate_negative_sample:
        negative_mappings = []

        for i, line in enumerate(conflict_mappings):
            mapping = line.strip().split("|")

            c1 = mapping[0]
            c2 = mapping[1]

            n1 = left_names[c1]
            n2 = right_names[c2]

            for l1, l2 in [(x, y) for x in n1 for y in n2]:
                origin = "-%d|%s|%s" % (i + 1, c1, c2)
                label = "-%d|%s|%s" % (i + 1, l1, l2)
                negative_mappings.append([origin, label])

        train_neg_mappings, valid_neg_mappings = split_train_valid(negative_mappings, train_rate=args.train_rate)

        train_mappings = (
            train_mappings * args.sample_duplicate
            + train_rv_mappings * args.sample_duplicate
            + train_neg_mappings
            #+ negative_sampling(pos_mappings=train_mappings,
            #                    left_names=left_names,
            #                    right_names=right_names)
        )
        validation_mappings = (
            validation_mappings * args.sample_duplicate
            + valid_rv_mappings * args.sample_duplicate
            + valid_neg_mappings
            #+ negative_sampling(pos_mappings=validation_mappings,
            #                    left_names=left_names,
            #                    right_names=right_names)
        )
    else:
        train_mappings = train_mappings + train_rv_mappings
        validation_mappings = validation_mappings + valid_rv_mappings

    with open("train_mappings.txt", "w") as f2:
        for m in train_mappings:
            f2.write(m[0] + "\n")
            f2.write(m[1] + "\n")

    with open("validation_mappings.txt", "w") as f2:
        for m in validation_mappings:
            f2.write(m[0] + "\n")
            f2.write(m[1] + "\n")
