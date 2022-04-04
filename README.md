# LogMap-ML

[![DOI](https://img.shields.io/badge/DOI-10.1007%2F978--3--030--77385--4__23-blue)](https://openaccess.city.ac.uk/id/eprint/25810/1/ESWC2021_ontology_alignment_LogMap_ML.pdf)

> LogMap-ML ships with a myriad of features, but I wanted to keep things simple. This is an **opinionated fork** with great defaults.

This repository includes the implementation of LogMap-ML introduced in the paper **Augmenting Ontology Alignment by Semantic Embedding and Distant Supervision** (see also [LogMap](https://github.com/ernestojimenezruiz/logmap-matcher/)).

The HeLis and FoodOn ontologies, and their partial GS, which are adopted for the evaluation in the paper, are under `eval/`.
Note the HeLis ontology adopted has been pre-processed by transforming instances into classes.

## Install 

Run:

```sh
$ python3 -m venv .venv

$ source .venv/bin/activate

$ python -m pip install -r requirements.txt

$ python -m spacy download en_core_web_sm
```

## Usage

### Pre-process: Run the original system

Run LogMap 4.0:

```sh
mkdir logmap_output_food

java -jar logmap/logmap-matcher-4.0.jar MATCHER \
  file:$(pwd)/use_cases/food/helis_v1.00.owl file:$(pwd)/use_cases/food/foodon-merged.owl $(pwd)/logmap_output/ true
```

This leads to LogMap initial set of candidate mappings or _anchors_
(<img src="https://render.githubusercontent.com/render/math?math=\mathcal{M}_a">)
and
over-estimation class mappings
(<img src="https://render.githubusercontent.com/render/math?math=\mathcal{M}_o">).

### Pre-process: Get Embedding Models

You can either download the word2vec embedding by gensim (the one trained with a corpus of Wikipedia articles from 2018-[download](https://drive.google.com/file/d/1rm9uJEKG25PJ79zxbZUWuaUroWeoWbFR/view?usp=sharing)) or use the ontology-tailored [OWL2Vec\*](https://github.com/KRR-Oxford/OWL2Vec-Star) embedding. 

The ontologies can use their own embedding models or use one common embedding model.

### Run Standalone

To use LogMap-ML as an standalone script, run:

```shell
python standalone.py --config default.cfg
```

LogMap-ML will extract the class name and path information for each class in both _to-be-aligned_ ontologies 
<img src="https://render.githubusercontent.com/render/math?math=\mathcal{O}_1">
and 
<img src="https://render.githubusercontent.com/render/math?math=\mathcal{O}_2">.

Then, it will generate high-confidence train mappings
(_seed mappings_ <img src="https://render.githubusercontent.com/render/math?math=\mathcal{M}_s">)
for training.<sup>1</sup>

LogMap-ML trains a Siamese Neural Network (SiamNN) as the mapping prediction model and compute the output mappings starting from a set of high recall candidate mappings (LogMap’s over-estimation mappings <img src="https://render.githubusercontent.com/render/math?math=\mathcal{M}_o">) to reduce the search space.

<sup>1</sup> A set of class disjointness constraints (branch conflicts) denoted
<img src="https://render.githubusercontent.com/render/math?math=\Delta">
are used to filter out some false-positive mappings from the LogMap anchor mappings.
When using class disjointness constraints to filter a mapping 
<img src="https://render.githubusercontent.com/render/math?math=m = (c_1,c_2) \in \mathcal{M}_a">
, we consider not just 
<img src="https://render.githubusercontent.com/render/math?math=c_1">
and 
<img src="https://render.githubusercontent.com/render/math?math=c_2">
, but all subsumers of 
<img src="https://render.githubusercontent.com/render/math?math=c_1">
and 
<img src="https://render.githubusercontent.com/render/math?math=c_2">
in the corresponding ontologies
<img src="https://render.githubusercontent.com/render/math?math=\mathcal{O}_1">
and 
<img src="https://render.githubusercontent.com/render/math?math=\mathcal{O}_2">.

### Evaluate

Assuming that gold standards (complete ground truth mappings) are given, Precision and Recall can be directly calculated by:

```sh
python evaluate.py --GS use_cases/food/reference.rdf --anchors logmap_output_food/logmap_anchors.txt --prediction prediction.txt
```

## Publications

* Jiaoyan Chen, Ernesto Jimenez-Ruiz, Ian Horrocks, Denvar Antonyrajah, Ali Hadian, Jaehun Lee. **Augmenting Ontology Alignment by Semantic Embedding and Distant Supervision**. European Semantic Web Conference, ESWC 2021. ([PDF](https://openaccess.city.ac.uk/id/eprint/25810/1/ESWC2021_ontology_alignment_LogMap_ML.pdf))

## TODO 

Best: Threshold: 1.00, precision: 0.720, recall: 0.602, f1: 0.656 EN FOOD
