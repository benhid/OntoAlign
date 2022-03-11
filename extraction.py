import argparse
import json
from pathlib import Path

from owlready2 import entity, owl, get_ontology

from lib.Label import label_to_string, name_to_string

# Initiate the parser.
parser = argparse.ArgumentParser()
parser.add_argument("onto", type=Path, help="Path to the ontology file.")


def super_classes(cls):
    supclasses = list()
    for supclass in cls.is_a:
        if type(supclass) == entity.ThingClass and supclass != owl.Thing:
            supclasses.append(supclass)
    return supclasses


def get_class_names(cls) -> list[str]:
    """
    Get the URI name and english labels of a class.
    """
    name = [name_to_string(cls.name)]
    labels = [label_to_string(label) for label in cls.label]
    #supclasses = [name_to_string(sup.name) for sup in super_classes(cls)]
    return name + labels


if __name__ == "__main__":
    # Read arguments from the command line.
    args = parser.parse_args()

    onto = get_ontology(str(args.onto)).load()

    c_names = {}

    for c in onto.classes():
        c_names[c.iri] = get_class_names(c)

    with open(args.onto.stem + "_names.json", "w") as outfile:
        json.dump(c_names, outfile)
