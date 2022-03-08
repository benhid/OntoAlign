import re
from typing import Dict, List, Optional

# Common namespaces.
namespaces = [
    "http://www.fbk.eu/ontologies/virtualcoach#",
    "http://purl.obolibrary.org/obo/",
    "http://bioontology.org/projects/ontologies/fma/fmaOwlDlComponent_2_0#",
    "http://www.ihtsdo.org/snomed#",
    "http://www.orpha.net/ORDO/",
    "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#",
]
prefixes = ["vc:", "obo:", "fma:", "snomed:", "ordo:", "nci:"]


def replace_with_prefix(uri: str) -> str:
    """
    Replaces known URI namespaces with prefixes.
    """
    for i, namespace in enumerate(namespaces):
        if namespace in uri:
            return uri.replace(namespace, prefixes[i])
    return uri


def parse_uri_name(uri: str) -> str:
    """
    Parse the URI name (camel cases).
    """
    uri = (
        uri.replace("_", " ")
        .replace("-", " ")
        .replace(".", " ")
        .replace("/", " ")
        .replace('"', " ")
        .replace("'", " ")
    )
    words = []
    for item in uri.split():
        matches = re.finditer(
            ".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", item
        )
        for m in matches:
            word = m.group(0)
            words.append(word.lower())
    return " ".join(words)


def entity_to_string(entity: str, names: Dict[str, List[str]]) -> str:
    """
    Transform an entity to string.
    """
    name = names[entity]
    # name[0] corresponds to URI name, meanwhile name[1] corresponds to the label (if any).
    if name[1] is None:
        uri_name = name[0]
        name_str = parse_uri_name(uri_name)
    else:
        label = name[1]
        name_str = label.lower().replace('"', "")
    return f'"{name_str}"'


def path_to_string(path: List[str], names: Dict[str, List[str]], keep_uri: bool) -> str:
    """
    Concatenate any number of entities in a path.
    Strings are quoted and concatenated with a comma inserted in between.
    """
    names_ = []
    for entity in path:
        if keep_uri:
            names_.append(f'"{entity}"')
        else:
            names_.append(entity_to_string(entity, names=names))
    return ",".join(names_)


def get_label(
    cls: str,
    names: Dict[str, List[str]],
    paths: Optional[List[List[str]]] = None,
    keep_uri: bool = False,
) -> str:
    if paths:
        # Search for the path with the class.
        for p in paths:
            # If found, convert and return path as string.
            if cls in p:
                path = p[p.index(cls) :]
                return path_to_string(path=path, names=names, keep_uri=keep_uri)
    else:
        return path_to_string(path=[cls], names=names, keep_uri=keep_uri)
    return '""'
