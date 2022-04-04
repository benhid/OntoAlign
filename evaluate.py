import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def read_oaei_mappings(file_path):
    tree = ET.parse(file_path)
    mappings_str = list()
    all_mappings_str = list()
    for t in tree.getroot():
        for m in t:
            if "map" in m.tag:
                for c in m:
                    mapping = list()
                    mv = "?"
                    for i, v in enumerate(c):
                        if i < 2:
                            for value in v.attrib.values():
                                mapping.append(value.lower())
                                break
                        if i == 3:
                            mv = v.text
                    all_mappings_str.append("|".join(mapping))
                    if not mv == "?":
                        mappings_str.append("|".join(mapping))
    return mappings_str, all_mappings_str


if __name__ == "__main__":
    # Initiate the parser.
    parser = argparse.ArgumentParser()
    parser.add_argument("--GS", type=str, help="Path to GS file.")
    parser.add_argument(
        "--anchors",
        type=Path,
        default=None,
        help="Path to the LogMap anchors file.",
    )
    parser.add_argument(
        "--prediction",
        type=str,
    )
    parser.add_argument("--threshold", type=float, default=0.9)

    # Read arguments from the command line.
    args = parser.parse_args()

    ref_mappings, ref_all_mappings = read_oaei_mappings(file_path=args.GS)
    ref_excluded_mappings = set(ref_all_mappings) - set(ref_mappings)

    print(f"{len(ref_mappings)} reference mappings")

    anchors = []
    if args.anchors:
        with open(args.anchors) as f:
            for line in f.readlines():
                tmp = line.strip().split("|")
                anchors.append("%s|%s" % (tmp[0].lower(), tmp[1].lower()))

        print(f"{len(anchors)} anchor mappings")

    prediction = []
    with open(args.prediction) as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.strip().split("|")
            pair = (tmp[1].lower(), tmp[2].lower())
            if float(tmp[-1]) >= args.threshold and pair not in prediction:
                prediction.append("%s|%s" % pair)

    # We include anchors in our predictions.
    for anchor in anchors:
        if anchor not in prediction:
            prediction.append(anchor)

    print(f"{len(prediction)} predictions")

    missed_mappings = []
    recall_num = 0
    for ref_mapping in ref_mappings:
        if ref_mapping in prediction:
            recall_num += 1
        else:
            missed_mappings.append(ref_mapping)

    print("%s missed mappings" % len(missed_mappings))

    R = recall_num / len(ref_mappings)

    precision_num = 0
    num = 0
    for s in prediction:
        if s not in ref_excluded_mappings:
            if s in ref_mappings:
                precision_num += 1
            num += 1
    P = precision_num / num

    F1 = 2 * P * R / (P + R)

    with open("missed_mappings.txt", "w") as outfile:
        for m in missed_mappings:
            outfile.write(m + "\n")

    print(
        "Threshold: %.2f, precision: %.3f, recall: %.3f, f1: %.3f"
        % (args.threshold, P, R, F1)
    )
