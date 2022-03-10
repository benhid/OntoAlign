import argparse
import xml.etree.ElementTree as ET

# Initiate the parser.
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--oaei_GS", type=str, help="Path to GS file.")
parser.add_argument(
    "--anchors",
    type=Path,
    default="logmap_output/logmap_anchors.txt",
    help="Path to the LogMap anchors file.",
)
parser.add_argument(
    "--prediction",
    type=str,
)
parser.add_argument("-t", "--threshold", type=float, default=0.9)


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
    # Read arguments from the command line.
    args = parser.parse_args()

    ref_mappings, ref_all_mappings = read_oaei_mappings(file_path=args.oaei_GS)
    ref_excluded_mappings = set(ref_all_mappings) - set(ref_mappings)

    anchors = list()
    with open(args.anchors) as f:
        for line in f.readlines():
            tmp = line.strip().split("|")
            anchors.append("%s|%s" % (tmp[0].lower(), tmp[1].lower()))

    prediction = list()
    with open(args.prediction) as f:
        lines = f.readlines()
        for j in range(0, len(lines), 3):
            tmp = lines[j].split("|")
            if float(tmp[4]) >= args.threshold:
                prediction.append("%s|%s" % (tmp[1].lower(), tmp[2].lower()))

    # Merge.
    for anchor in anchors:
        if anchor not in prediction:
            prediction.append(anchor)

    recall_num = 0
    for ref_mapping in ref_mappings:
        if ref_mapping in prediction:
            recall_num += 1
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

    print(
        "%d mappings, Precision: %.3f, Recall: %.3f, F1: %.3f"
        % (len(prediction), P, R, F1)
    )
