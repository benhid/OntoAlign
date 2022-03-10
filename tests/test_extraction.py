from owlready2 import Thing, get_ontology

from extraction import get_class_names, get_class_path

onto = get_ontology("http://test.org/onto.owl")

with onto:

    class Dummy(Thing):
        pass


def test_get_class_names_from_class():
    cls = Dummy("my_class")
    names = get_class_names(cls)
    assert "my_class" == names[0]
    assert names[1] is None


def test_get_class_names_from_class_with_labels():
    cls = Dummy("my_class")
    cls.label = ["One", "Two", "Three"]
    names = get_class_names(cls)
    assert "my_class" == names[0]
    assert "One" == names[1]


def test_get_class_paths_from_class():
    cls = Dummy("my_class")
    cls.label = ["One", "Two"]
    paths = get_class_path(cls)
    assert "http://test.org/onto.owl#my_class" == paths[0]
    assert "http://test.org/onto.owl#Dummy" == paths[1]
