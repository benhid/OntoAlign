from sample import negative_sampling, violates_branch_conflict


def test_violates_branch_conflict():
    l1 = '"basic food","food"'
    l2 = '"organism","material entity"'
    assert violates_branch_conflict(l1, l2)


def test_negative_sampling():
    mappings = [["1|origin|vc:Food|vc:Meal", '1|name|"food"|"meal","meal"']]
    left_paths = [["vc:BasicFood", "vc:Food"], ["vc:Food"]]
    right_paths = [["vc:ConsumedFood"], ["vc:Meal"]]
    left_names = {
        "vc:Food": ["Food", None],
        "vc:BasicFood": ["BasicFood", "Basic Food"],
    }
    right_names = {
        "vc:Meal": ["Meal", "Meal"],
        "vc:ConsumedFood": ["ConsumedFood", "Consumed Food"],
    }
    neg_mappings = negative_sampling(
        mappings=mappings,
        left_paths=left_paths,
        right_paths=right_paths,
        left_names=left_names,
        right_names=right_names,
        keep_uri=False,
    )
    assert [
        "neg-1-h|origin|vc:BasicFood|vc:Meal",
        'neg-1-h|name|"basic food","food"|"meal","meal"',
    ] == neg_mappings[0]
    assert [
        "neg-1-f|origin|vc:Food|vc:ConsumedFood",
        'neg-1-f|name|"food"|"consumed food"',
    ] == neg_mappings[1]
