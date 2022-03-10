import re
from typing import Optional


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


def entity_to_string(entity: str, names: dict[str, tuple[str, Optional[str]]]) -> str:
    """
    Transform an entity to string.
    """
    name = names[entity]
    # Label is preferred over URI name.
    if name[1] is None:
        uri_name = name[0]
        name_str = parse_uri_name(uri_name)
    else:
        label = name[1]
        name_str = label.lower().replace('"', "")
    return f'"{name_str}"'


def path_to_string(
    path: list[str], names: dict[str, tuple[str, Optional[str]]], keep_uri: bool
) -> str:
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
    names: dict[str, tuple[str, Optional[str]]],
    paths: Optional[list[list[str]]] = None,
    keep_uri: bool = False,
) -> str:
    # if label_type == 'path':
    if paths:
        # Search for the path with the class.
        for p in paths:
            # If found, convert and return path as string.
            if cls in p:
                path = p[p.index(cls) :]
                return path_to_string(path=path, names=names, keep_uri=keep_uri)
    else:
        return path_to_string(path=[cls], names=names, keep_uri=keep_uri)
    # This could happen if the class is not in paths.
    return '""'
