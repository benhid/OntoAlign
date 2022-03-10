# LogMap-ML

[![DOI](https://img.shields.io/badge/DOI-10.1007%2F978--3--030--77385--4__23-blue)](https://openaccess.city.ac.uk/id/eprint/25810/1/ESWC2021_ontology_alignment_LogMap_ML.pdf)

> LogMap-ML ships with a myriad of features, but I wanted to keep things simple. This is an **opinionated fork** with great defaults.

This repository includes the implementation of LogMap-ML introduced in the paper **Augmenting Ontology Alignment by Semantic Embedding and Distant Supervision** (see also [LogMap](https://github.com/ernestojimenezruiz/logmap-matcher/)).

The HeLis and FoodOn ontologies, and their partial GS, which are adopted for the evaluation in the paper, are under `data/`.
Note the HeLis ontology adopted has been pre-processed by transforming instances into classes.

## Install 

Run:

```sh
$ python3 -m venv venv

$ source venv/bin/activate

$ python -m pip install -r requirements.txt
```

## Usage

### Pre-process #1: Run the original system

Run LogMap 4.0:

```sh
$ java -jar logmap/logmap-matcher-4.0.jar MATCHER file:/path/to/data/helis_v1.00.owl file:/path/to/data/foodon-merged.owl /path/to/data/logmap_output/ true
```

This leads to LogMap initial set of candidate mappings or _anchors_
(<img src="https://render.githubusercontent.com/render/math?math=\mathcal{M}_a">)
and
over-estimation class mappings
(<img src="https://render.githubusercontent.com/render/math?math=\mathcal{M}_o">).

### Pre-process #2: Get Embedding Models

You can either download the word2vec embedding by gensim (the one trained with a corpus of Wikipedia articles from 2018-[download](https://drive.google.com/file/d/1rm9uJEKG25PJ79zxbZUWuaUroWeoWbFR/view?usp=sharing)) or use the ontology-tailored [OWL2Vec\*](https://github.com/KRR-Oxford/OWL2Vec-Star) embedding. 

The ontologies can use their own embedding models or use one common embedding model.

### Pre-process #3: Class Name and Path Extraction

This is to extract the class name and path information for each class in an ontology:

```sh
$ python extraction.py data/cmt.owl
```

It should be executed separately for both _to-be-aligned_ ontologies 
<img src="https://render.githubusercontent.com/render/math?math=\mathcal{O}_1">
and 
<img src="https://render.githubusercontent.com/render/math?math=\mathcal{O}_2">.

### Step #1: Sample

This is to generate high-confidence train mappings
(<img src="https://render.githubusercontent.com/render/math?math=\mathcal{M}_s">)
for training:

```sh
$ python sample.py data/logmap_output/logmap_anchors.txt --left_names data/cmt_names.json --left_paths data/cmt_paths.txt --right_names data/sigkdd_names.json --right_paths data/sigkdd_paths.txt
```

A set of class disjointness constraints (branch conflicts) denoted
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

### Step #2: Train, Valid and Predict

We train a Siamese Neural Network (SiamNN) as the mapping prediction model:

```sh
$ python train_valid.py --left_w2v enwiki_model/word2vec_gensim --right_w2v enwiki_model/word2vec_gensim --train_mappings data/train_mappings.txt --valid_mappings data/validation_mappings.txt --nn_dir data/model
```

Finally, we can compute the output mappings starting from a set of high recall candidate mappings (LogMap’s over-estimation mappings) to reduce the search space:

```sh
$ python predict_candidates.py data/logmap_output/logmap_overestimation.txt --left_w2v enwiki_model/word2vec_gensim --right_w2v enwiki_model/word2vec_gensim --left_names data/cmt_names.json --left_paths data/cmt_paths.txt --right_names data/sigkdd_names.json --right_paths data/sigkdd_paths.txt --nn_dir data/model
```

### Step #3: Evaluate

Assuming that gold standards (complete ground truth mappings) are given, Precision and Recall can be directly calculated by:

```sh
$ python evaluate.py --oaei_GS data/cmt-sigkdd.rdf --anchors data/logmap_output/logmap_anchors.txt --prediction data/prediction.txt
```

## Publications

* Jiaoyan Chen, Ernesto Jimenez-Ruiz, Ian Horrocks, Denvar Antonyrajah, Ali Hadian, Jaehun Lee. **Augmenting Ontology Alignment by Semantic Embedding and Distant Supervision**. European Semantic Web Conference, ESWC 2021. ([PDF](https://openaccess.city.ac.uk/id/eprint/25810/1/ESWC2021_ontology_alignment_LogMap_ML.pdf))
