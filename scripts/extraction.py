import argparse
import json
from pathlib import Path
from typing import List

from lib.Label import replace_with_prefix
from owlready2 import *

# Initiate the parser.
parser = argparse.ArgumentParser()
parser.add_argument("onto", type=Path, help="Path to the ontology file.")

# Read arguments from the command line.
args = parser.parse_args()


def get_class_names(cls) -> List[str]:
    """
    Get the URI name and english label for each class.
    """
    name = cls.name
    labels = cls.label.en + cls.label  # concatenate lists
    return [name, labels[0] if labels else None]


def super_classes(cls) -> List[EntityClass]:
    sc = []
    for sup_class in cls.is_a:
        if type(sup_class) == entity.ThingClass:
            sc.append(sup_class)
    return sc


def append_super_class(cls, p):
    """
    Recursive function.
    """
    p.append(replace_with_prefix(uri=cls.iri))
    sup_classes = super_classes(cls=cls)
    if owl.Thing in sup_classes or not sup_classes:
        return p
    else:
        # As one class may have multiple paths, we randomly select one. TODO - Verify.
        return append_super_class(cls=sup_classes[0], p=p)


def get_class_path(cls):
    """
    Get the sequences of classes obtained by traversing the class hierarchy back from class to owl:Thing
    """
    return append_super_class(cls=cls, p=list())


if __name__ == "__main__":
    onto = get_ontology(str(args.onto)).load()

    c_names = {}
    c_paths = []

    for c in onto.classes():
        c_names[replace_with_prefix(uri=c.iri)] = get_class_names(c)
        c_paths.append(get_class_path(c))

    with open(args.onto.stem + "_names.json", "w") as outfile:
        json.dump(c_names, outfile)

    with open(args.onto.stem + "_paths.txt", "w") as f:
        for path in c_paths:
            f.write("%s\n" % ",".join(path))
