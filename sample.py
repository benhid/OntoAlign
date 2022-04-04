import argparse
import json
import random
from pathlib import Path

"""
# Class disjointness constraints for OAEI Conference Track.
branch_conflicts = [
    ["http://conference#Regular_contribution", "http://ekaw#Paper"],
    ["http://conference#Conference_document", "http://ekaw#Event"],
    ["http://conference#Conference_document", "http://iasted#Conference_activity"],
    ["http://edas#Author", "http://ekaw#Conference_Participant"],
    ["http://ekaw#Possible_Reviewer", "http://iasted#Speaker"],
    ["http://cmt#Reviewer", "http://ekaw#Possible_Reviewer"],
    ["http://conference#Conference_document", "http://edas#ReviewRating"],
    ["http://edas#Conference", "http://ekaw#Event"],
    ["http://edas#ConferenceSession", "http://ekaw#Event"],
    ["http://conference#Conference_document", "http://iasted#Person"],
    ["http://conference#Conference_document", "http://ekaw#Person"],
    ["http://conference#Conference", "http://ekaw#Event"],
    ["http://conference#Conference", "http://confOf#Event"],
    ["http://iasted#Money", "http://sigkdd#Sponzor"],
    ["http://confOf#Event", "http://edas#Conference"],
    ["http://cmt#Document", "http://edas#ReviewRating"],
    ["http://conference#Conference_document", "http://iasted#State"],
    ["http://ekaw#Document", "http://iasted#Person"],
    ["http://conference#Conference", "http://edas#Conference"],
    ["http://edas#ConferenceEvent", "http://ekaw#Document"],
    ["http://edas#ConferenceSession", "http://ekaw#Document"],
    ["http://edas#ConferenceEvent", "http://iasted#Place"],
    ["http://ekaw#Event", "http://iasted#City"],
]
"""
# Class disjointness constraints for `food` use case.
branch_conflicts = [
    [
        "http://www.fbk.eu/ontologies/virtualcoach#FOOD-1318",
        "http://purl.obolibrary.org/obo/NCBITaxon_52904",
    ],
    [
        "http://www.fbk.eu/ontologies/virtualcoach#FOOD-8022",
        "http://purl.obolibrary.org/obo/NCBITaxon_2769",
    ],
    [
        "http://www.fbk.eu/ontologies/virtualcoach#FOOD-1314",
        "http://purl.obolibrary.org/obo/NCBITaxon_6563",
    ],
    [
        "http://www.fbk.eu/ontologies/virtualcoach#VitaminB12",
        "http://purl.obolibrary.org/obo/FOODON_03413761",
    ],
    [
        "http://www.fbk.eu/ontologies/virtualcoach#FOOD-1322",
        "http://purl.obolibrary.org/obo/NCBITaxon_30948",
    ],
    [
        "http://www.fbk.eu/ontologies/virtualcoach#FOOD-306",
        "http://purl.obolibrary.org/obo/FOODON_03411309",
    ],
    [
        "http://www.fbk.eu/ontologies/virtualcoach#FOOD-1319",
        "http://purl.obolibrary.org/obo/NCBITaxon_27697",
    ],
    [
        "http://www.fbk.eu/ontologies/virtualcoach#Salt",
        "http://purl.obolibrary.org/obo/CHEBI_24866",
    ],
    [
        "http://www.fbk.eu/ontologies/virtualcoach#FOOD-1313",
        "http://purl.obolibrary.org/obo/NCBITaxon_8043",
    ],
    [
        "http://www.fbk.eu/ontologies/virtualcoach#Mollusc",
        "http://purl.obolibrary.org/obo/NCBITaxon_6447",
    ],
    [
        "http://www.fbk.eu/ontologies/virtualcoach#FOOD-8029",
        "http://purl.obolibrary.org/obo/NCBITaxon_55118",
    ],
    [
        "http://www.fbk.eu/ontologies/virtualcoach#FOOD-666006",
        "http://purl.obolibrary.org/obo/CHEBI_2877",
    ],
    [
        "http://www.fbk.eu/ontologies/virtualcoach#FOOD-8022",
        "http://purl.obolibrary.org/obo/FOODON_03411742",
    ],
    [
        "http://www.fbk.eu/ontologies/virtualcoach#FOOD-1317",
        "http://purl.obolibrary.org/obo/NCBITaxon_117893",
    ],
    [
        "http://www.fbk.eu/ontologies/virtualcoach#FOOD-1325",
        "http://purl.obolibrary.org/obo/NCBITaxon_1489894",
    ],
    [
        "http://www.fbk.eu/ontologies/virtualcoach#FOOD-100143",
        "http://purl.obolibrary.org/obo/CHEBI_28017",
    ],
    [
        "http://www.fbk.eu/ontologies/virtualcoach#Algae",
        "http://purl.obolibrary.org/obo/FOODON_03411301",
    ],
    [
        "http://www.fbk.eu/ontologies/virtualcoach#FOOD-1311",
        "http://purl.obolibrary.org/obo/NCBITaxon_8008",
    ],
    [
        "http://www.fbk.eu/ontologies/virtualcoach#Starch",
        "http://purl.obolibrary.org/obo/FOODON_03301000",
    ],
    [
        "http://www.fbk.eu/ontologies/virtualcoach#Mushrooms",
        "http://purl.obolibrary.org/obo/FOODON_00001287",
    ],
    [
        "http://www.fbk.eu/ontologies/virtualcoach#FOOD-1300",
        "http://purl.obolibrary.org/obo/NCBITaxon_43062",
    ],
    [
        "http://www.fbk.eu/ontologies/virtualcoach#FOOD-327",
        "http://purl.obolibrary.org/obo/FOODON_03414537",
    ],
    [
        "http://www.fbk.eu/ontologies/virtualcoach#FOOD-326",
        "http://purl.obolibrary.org/obo/FOODON_03411171",
    ],
    [
        "http://www.fbk.eu/ontologies/virtualcoach#FOOD-1320",
        "http://purl.obolibrary.org/obo/NCBITaxon_6608",
    ],
]


def violates_branch_conflict(c1: str, c2: str) -> bool:
    for conflict in branch_conflicts:
        if conflict[0] == c1 and conflict[1] == c2:
            return True
    return False


def train_valid_split(mappings: list, train_rate: float):
    random.shuffle(mappings)
    ratio = round(len(mappings) * train_rate)
    train = mappings[0:ratio]
    if train_rate < 1.0:
        valid = mappings[ratio:]
    else:
        valid = mappings[round(len(mappings) * 0.8) :]
    return train, valid


def negative_sampling(pos_mappings, mappings, left_names, right_names):
    # Note that the random replacements could produce positive samples from Ms;
    # we discard any such negative samples.
    pos_classes = []

    for line in pos_mappings:
        mapping = line.strip().split("|")
        c1, c2 = mapping[1:3]
        pos_classes.append((c1, c2))

    # Generate random negative mappings.
    neg_mappings = []

    for line in mappings:
        mapping = line.strip().split("|")

        idx, c1, c2 = mapping[0:3]

        n1 = left_names.get(c1)
        n2 = right_names.get(c2)

        neg_c2 = random.sample(list(right_names.keys() - {c2}), 1)[0]
        neg_n2 = right_names.get(neg_c2)

        if (c1, neg_c2) not in pos_classes:
            if n1 and neg_n2:
                for l1, l2 in [(x, y) for x in n1 for y in neg_n2]:
                    m = f"-{idx}-f|{c1}|{neg_c2}|{l1}|{l2}"
                    neg_mappings.append(m)

        neg_c1 = random.sample(list(left_names.keys() - {c1}), 1)[0]
        neg_n1 = left_names.get(neg_c1)

        if (neg_c1, c2) not in pos_classes:
            if neg_n1 and n2:
                for l1, l2 in [(x, y) for x in neg_n1 for y in n2]:
                    m = f"-{idx}-h|{c1}|{neg_c2}|{l1}|{l2}"
                    neg_mappings.append(m)

    return neg_mappings


def logmap_sampling(mappings, left_names, right_names):
    pos_mappings = []
    rule_violated_mappings = []

    for i, line in enumerate(mappings):
        mapping = line.strip().split("|")

        c1, c2 = mapping[0:2]

        n1 = left_names.get(c1)
        n2 = right_names.get(c2)

        if n1 and n2:
            for l1, l2 in [(x, y) for x in n1 for y in n2]:
                if violates_branch_conflict(c1, c2):
                    m = f"-{i + 1}|{c1}|{c2}|{l1}|{l2}"
                    rule_violated_mappings.append(m)
                else:
                    m = f"{i + 1}|{c1}|{c2}|{l1}|{l2}"
                    pos_mappings.append(m)

    return pos_mappings, rule_violated_mappings


if __name__ == "__main__":
    # Initiate the parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "anchors",
        type=Path,
        default="logmap_output/logmap_anchors.txt",
        help="Path to LogMap anchors file.",
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
        "--train-rate",
        type=float,
        default=1.0,
        help="Can be set to 1.0 (to use all the seeds as the training set and 20% of them as the validation set)"
        "or a float smaller than 1.0 (where train_rate of all the samples are used as the training set and the remaining are used as the validation set).",
    )
    parser.add_argument(
        "--augment-negative_sample",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--sample_duplicate", type=int, default=2)

    # Read arguments from the command line.
    args = parser.parse_args()

    # Read initial set of candidate mappings (anchors).
    with open(args.anchors, "r") as infile:
        anchors = infile.readlines()

    # Generate mappings from LogMap anchors.
    with open(args.left_names, "r") as infile:
        left_names = json.load(infile)

    with open(args.right_names, "r") as infile:
        right_names = json.load(infile)

    positive_mappings, rule_violated_mappings = logmap_sampling(
        anchors, left_names, right_names
    )

    print(
        f"{len(positive_mappings)} positive mappings, {len(rule_violated_mappings)} violate the rules"
    )

    train_mappings, validation_mappings = train_valid_split(
        positive_mappings, train_rate=args.train_rate
    )

    print(f"{len(train_mappings)} train, {len(validation_mappings)} validation")

    # Adopt anchor mappings that violate the class disjointness constraints as negative samples and
    # randomly partition them into a training set and a validation set with the same ratio:
    train_rv_mappings, valid_rv_mappings = train_valid_split(
        rule_violated_mappings, train_rate=args.train_rate
    )

    print(f"{len(train_rv_mappings)} train rv, {len(valid_rv_mappings)} validation rv")

    # We want to apply data augmentation on *only* the training set.
    if args.augment_negative_sample:
        train_mappings = (
            train_mappings * args.sample_duplicate
            + train_rv_mappings * args.sample_duplicate
            + negative_sampling(
                positive_mappings, train_mappings, left_names, right_names
            )
        )
    else:
        train_mappings = train_mappings + train_rv_mappings
    validation_mappings = validation_mappings + valid_rv_mappings

    print(f"All - {len(train_mappings)} train, {len(validation_mappings)} validation ")

    with open("train_mappings.txt", "w") as outfile:
        for m in train_mappings:
            outfile.write(m + "\n")

    with open("validation_mappings.txt", "w") as outfile:
        for m in validation_mappings:
            outfile.write(m + "\n")
