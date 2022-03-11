# LogMap-ML

[![DOI](https://img.shields.io/badge/DOI-10.1007%2F978--3--030--77385--4__23-blue)](https://openaccess.city.ac.uk/id/eprint/25810/1/ESWC2021_ontology_alignment_LogMap_ML.pdf)

> LogMap-ML ships with a myriad of features, but I wanted to keep things simple. This is an **opinionated fork** with great defaults.

This repository includes the implementation of LogMap-ML introduced in the paper **Augmenting Ontology Alignment by Semantic Embedding and Distant Supervision** (see also [LogMap](https://github.com/ernestojimenezruiz/logmap-matcher/)).

The HeLis and FoodOn ontologies, and their partial GS, which are adopted for the evaluation in the paper, are under `eval/`.
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
mkdir logmap_output

java -jar logmap/logmap-matcher-4.0.jar MATCHER \
  file:$(pwd)/use_cases/oaei_2021/cmt.owl file:$(pwd)/use_cases/oaei_2021/sigkdd.owl $(pwd)/logmap_output/ true

# PAPER

mkdir logmap_output

java -jar logmap/logmap-matcher-4.0.jar MATCHER \
  file:$(pwd)/use_cases/food/helis_v1.00.owl file:$(pwd)/use_cases/food/foodon-merged.owl $(pwd)/logmap_output/ true
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
python extraction.py use_cases/oaei_2021/cmt.owl
python extraction.py use_cases/oaei_2021/sigkdd.owl

# PAPER
python extraction.py use_cases/food/helis_v1.00.owl
python extraction.py use_cases/food/foodon-merged.owl
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
python sample.py logmap_output/logmap_anchors.txt \
  --conflicting_mappings logmap_output/logmap_logically_conflicting_mappings.txt \
  --left_names cmt_names.json --right_names sigkdd_names.json
# PAPER
python sample.py logmap_output/logmap_anchors.txt \
  --conflicting_mappings logmap_output/logmap_discarded_mappings.txt \
  --left_names helis_v1.00_names.json --right_names foodon-merged_names.json
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

```shell
python train_valid.py --train_mappings train_mappings.txt --valid_mappings validation_mappings.txt \
  --left_w2v owl2vec_model/owl2vec_cmt --right_w2v owl2vec_model/owl2vec_sigkdd --nn_dir model
python train_valid.py --train_mappings train_mappings.txt --valid_mappings validation_mappings.txt \
  --left_w2v enwiki_model/word2vec_gensim --right_w2v enwiki_model/word2vec_gensim --nn_dir model
# PAPER
python train_valid.py --train_mappings train_mappings.txt --valid_mappings validation_mappings.txt \
  --left_w2v owl2vec_model/owl2vec_helis_v1.00/output --right_w2v owl2vec_model/owl2vec_foodon-merged/output --nn_dir model
python train_valid.py --train_mappings train_mappings.txt --valid_mappings validation_mappings.txt \
  --left_w2v owl2vec_model/owl2vec_helis_v1.00_foodon-merged/output --right_w2v owl2vec_model/owl2vec_helis_v1.00_foodon-merged/output --nn_dir model
```

Finally, we can compute the output mappings starting from a set of high recall candidate mappings (LogMap’s over-estimation mappings) to reduce the search space:

```sh
python predict_candidates.py logmap_output/logmap_overestimation.txt \
  --left_w2v owl2vec_model/owl2vec_cmt --right_w2v owl2vec_model/owl2vec_sigkdd \
  --left_names cmt_names.json --right_names sigkdd_names.json --nn_dir model
python predict_candidates.py logmap_output/logmap_overestimation.txt \
  --left_w2v enwiki_model/word2vec_gensim --right_w2v enwiki_model/word2vec_gensim \
  --left_names cmt_names.json --right_names sigkdd_names.json --nn_dir model
# PAPER
python predict_candidates.py logmap_output/logmap_overestimation.txt \
  --left_w2v owl2vec_model/owl2vec_helis_v1.00/output --right_w2v owl2vec_model/owl2vec_foodon-merged/output \
  --left_names helis_v1.00_names.json --right_names foodon-merged_names.json --nn_dir model
python predict_candidates.py logmap_output/logmap_overestimation.txt \
  --left_w2v owl2vec_model/owl2vec_helis_v1.00_foodon-merged/output --right_w2v owl2vec_model/owl2vec_helis_v1.00_foodon-merged/output \
  --left_names helis_v1.00_names.json --right_names foodon-merged_names.json --nn_dir model
```

### Step #3: Evaluate

Assuming that gold standards (complete ground truth mappings) are given, Precision and Recall can be directly calculated by:

```sh
python evaluate.py --GS use_cases/oaei_2021/reference.rdf --anchors logmap_output/logmap_anchors.txt --prediction prediction.txt
# PAPER
python evaluate.py --GS use_cases/food/reference.rdf --anchors logmap_output/logmap_anchors.txt --prediction prediction.txt
```

## Publications

* Jiaoyan Chen, Ernesto Jimenez-Ruiz, Ian Horrocks, Denvar Antonyrajah, Ali Hadian, Jaehun Lee. **Augmenting Ontology Alignment by Semantic Embedding and Distant Supervision**. European Semantic Web Conference, ESWC 2021. ([PDF](https://openaccess.city.ac.uk/id/eprint/25810/1/ESWC2021_ontology_alignment_LogMap_ML.pdf))
