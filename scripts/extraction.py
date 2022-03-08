import argparse
import json
from pathlib import Path
from typing import Dict, List

from lib.Label import replace_with_prefix
from owlready2 import *

# Initiate the parser.
parser = argparse.ArgumentParser()
parser.add_argument("onto", type=Path, help="Path to the ontology file.")

# Read arguments from the command line.
args = parser.parse_args()


def get_class_names(cls) -> List[str]:
    """
    Get the URI name and english label (rdf:label) for each class.
    """
    name = cls.name
    labels = c.label.en + c.label  # `c.label.en` and `c.label` are both Python lists
    return [name, labels[0] if labels else None]


def super_classes(c):
    sc = list()
    for sup_class in c.is_a:
        if type(sup_class) == entity.ThingClass:
            sc.append(sup_class)
    return sc


def append_super_class(c, p):
    """
    Recursive function.
    """
    p.append(replace_with_prefix(uri=c.iri))
    sup_classes = super_classes(c=c)
    if owl.Thing in sup_classes or not sup_classes:
        return p
    else:
        return append_super_class(c=sup_classes[0], p=p)  # TODO descartamos el resto??


def get_class_path(cls):
    """
    Get path from each class to root.
    """
    return append_super_class(c=cls, p=list())


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
