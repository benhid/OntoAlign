import argparse
import configparser
import json
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec

from extraction import get_class_names, get_class_path, get_classes
from lib.Encoder import load_samples, to_samples
from lib.SiameseNetwork import siamese_nn_predict
from sample import logmap_sampling, negative_sampling, train_valid_split
from train_valid import train, valid

if __name__ == "__main__":
    # Initiate the parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=Path,
        default="default.cfg",
        help="Configuration file.",
    )

    # Read arguments from the command line.
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    # All intermediary files go in the cache folder.
    cache_dir = Path(config["default"]["cache_dir"])
    cache_dir.mkdir(exist_ok=True, parents=True)

    # Phase 1. Extraction
    classes = get_classes(Path(config["default"]["left_ontology"]))
    l_names = {}
    l_paths = []

    for cls in classes:
        l_names[cls.iri] = get_class_names(cls=cls)
        l_paths.append(get_class_path(cls=cls, p=list()))

    with open(cache_dir / "l_names.json", "w") as outfile:
        json.dump(l_names, outfile)

    with open(cache_dir / "l_paths.txt", "w") as outfile:
        for path in l_paths:
            outfile.write(",".join(path) + "\n")

    classes = get_classes(Path(config["default"]["right_ontology"]))
    r_names = {}
    r_paths = []

    for cls in classes:
        r_names[cls.iri] = get_class_names(cls=cls)
        r_paths.append(get_class_path(cls=cls, p=list()))

    with open(cache_dir / "r_names.json", "w") as outfile:
        json.dump(r_names, outfile)

    with open(cache_dir / "r_paths.txt", "w") as outfile:
        for path in r_paths:
            outfile.write(",".join(path) + "\n")

    # Phase 2. Sample
    # Read initial set of candidate mappings (anchors).
    with open(
        Path(config["default"]["logmap_output_dir"]) / "logmap_anchors.txt", "r"
    ) as infile:
        anchors = infile.readlines()

    # Generate mappings from LogMap anchors.
    positive_mappings, rule_violated_mappings = logmap_sampling(
        anchors, l_names, r_names
    )

    with open(cache_dir / "positive_mappings.txt", "w") as outfile:
        for m in positive_mappings:
            outfile.write(m + "\n")

    with open(cache_dir / "rule_violated_mappings.txt", "w") as outfile:
        for m in rule_violated_mappings:
            outfile.write(m + "\n")

    print(
        f"{len(positive_mappings)} positive mappings, {len(rule_violated_mappings)} violate the rules"
    )

    train_rate = float(config["train"]["train_rate"])

    train_mappings, validation_mappings = train_valid_split(
        positive_mappings, train_rate=train_rate
    )

    print(f"{len(train_mappings)} train, {len(validation_mappings)} validation")

    # Adopt anchor mappings that violate the class disjointness constraints as negative samples and
    # randomly partition them into a training set and a validation set with the same ratio:
    train_rv_mappings, valid_rv_mappings = train_valid_split(
        rule_violated_mappings, train_rate=train_rate
    )

    print(f"{len(train_rv_mappings)} rv train, {len(valid_rv_mappings)} rv validation")

    # We want to apply data augmentation on *only* the training set.
    if config["train"]["augment_negative_sample"] == "true":
        train_mappings = (
            train_mappings * int(config["train"]["sample_duplicate"])
            + train_rv_mappings * int(config["train"]["sample_duplicate"])
            + negative_sampling(positive_mappings, train_mappings, l_names, r_names)
        )
    else:
        train_mappings = train_mappings + train_rv_mappings
    validation_mappings = validation_mappings + valid_rv_mappings

    with open(cache_dir / "train_mappings.txt", "w") as outfile:
        for m in train_mappings:
            outfile.write(m + "\n")

    with open(cache_dir / "valid_mappings.txt", "w") as outfile:
        for m in validation_mappings:
            outfile.write(m + "\n")

    print(f"All - {len(train_mappings)} train, {len(validation_mappings)} validation")

    # Phase 3. Training.
    # Load models.
    left_owl2vec_model = Word2Vec.load(config["model"]["left_owl2vec"])
    right_owl2vec_model = Word2Vec.load(config["model"]["right_owl2vec"])

    left_wv_model = Word2Vec.load(config["model"]["left_word2vec"])
    right_wv_model = Word2Vec.load(config["model"]["right_word2vec"])

    print(f"Start train")

    encoder_type = config["nn"]["encoder_type"]

    train_X1, train_X2, train_Y, train_num = load_samples(
        mappings=train_mappings,
        left_owl2vec_model=left_owl2vec_model,
        right_owl2vec_model=right_owl2vec_model,
        left_wv_model=left_wv_model,
        right_wv_model=right_wv_model,
        left_tensors={},
        right_tensors={},
        encoder_type=encoder_type,
    )

    shuffle_indices = np.random.permutation(np.arange(train_num))
    train_X1, train_X2, train_Y = (
        train_X1[shuffle_indices],
        train_X2[shuffle_indices],
        train_Y[shuffle_indices],
    )

    nn_dir = cache_dir / config["nn"]["model_dir"]

    train(
        X1=train_X1,
        X2=train_X2,
        Y=train_Y,
        nn_dir=nn_dir,
    )

    valid_X1, valid_X2, valid_Y, valid_num = load_samples(
        mappings=validation_mappings,
        left_owl2vec_model=left_owl2vec_model,
        right_owl2vec_model=right_owl2vec_model,
        left_wv_model=left_wv_model,
        right_wv_model=right_wv_model,
        left_tensors={},
        right_tensors={},
        encoder_type=encoder_type,
    )

    threshold, f1, p, r, acc = valid(
        X1=valid_X1,
        X2=valid_X2,
        Y=valid_Y,
        nn_dir=nn_dir,
        valid_num=valid_num,
    )

    print(
        f"Best setting: Threshold: {threshold}, precision: {p}, recall: {r}, f1: {f1}, acc: {acc}\n"
    )

    # Phase 4. Predict candidates.
    with open(
        Path(config["default"]["logmap_output_dir"]) / "logmap_overestimation.txt", "r"
    ) as infile:
        candidates = infile.readlines()

    candidate_mappings = []

    for i, line in enumerate(candidates):
        mapping = line.strip().split("|")

        c1, c2 = mapping[0:2]

        n1 = l_names.get(c1)
        n2 = r_names.get(c2)

        if n1 and n2:
            for l1, l2 in [(x, y) for x in n1 for y in n2]:
                m = f"{i + 1}|{c1}|{c2}|{l1}|{l2}"
                candidate_mappings.append(m)

    print("%d candidates" % len(candidate_mappings))

    X1, X2 = to_samples(
        mappings=candidate_mappings,
        left_owl2vec_model=left_owl2vec_model,
        right_owl2vec_model=right_owl2vec_model,
        left_wv_model=left_wv_model,
        right_wv_model=right_wv_model,
        left_tensors={},
        right_tensors={},
        encoder_type=encoder_type,
    )

    test_distances = siamese_nn_predict(test_x1=X1, test_x2=X2, nn_dir=nn_dir)
    test_scores = 1 - test_distances

    with open(cache_dir / "prediction.txt", "w") as f:
        for i, mapping in enumerate(candidate_mappings):
            f.write("%s|%.3f\n" % (mapping, test_scores[i]))

    print(
        """
        python evaluate.py --GS use_cases/food/reference.rdf --prediction cache/prediction.txt --threshold 0.9 --anchors logmap_output_food/logmap_anchors.txt
        """
    )
