from lib.Label import get_label


def test_get_label_from_class_with_paths():
    cls = "vc:BasicFood"
    names = {
        "vc:CookedFood": ["CookedFood", None],
        "vc:BasicFood": ["BasicFood", "Basic Food"],
    }
    paths = [["vc:BasicFood", "vc:CookedFood"], ["vc:CookedFood"]]
    label = get_label(cls=cls, names=names, paths=paths, keep_uri=False)
    assert label == '"basic food","cooked food"'


def test_get_label_from_class_with_paths_and_uri():
    cls = "vc:BasicFood"
    names = {
        "vc:CookedFood": ["CookedFood", None],
        "vc:BasicFood": ["BasicFood", "Basic Food"],
    }
    paths = [["vc:BasicFood", "vc:CookedFood"], ["vc:CookedFood"]]
    label = get_label(cls=cls, names=names, paths=paths, keep_uri=True)
    assert label == '"vc:BasicFood","vc:CookedFood"'


def test_get_label_from_class():
    cls = "vc:BasicFood"
    names = {
        "vc:CookedFood": ["CookedFood", None],
        "vc:BasicFood": ["BasicFood", "Basic Food"],
    }
    label = get_label(cls=cls, names=names, keep_uri=False)
    assert label == '"basic food"'
