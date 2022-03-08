# LogMap-ML

[![DOI](https://img.shields.io/badge/DOI-10.1007%2F978--3--030--77385--4__23-blue)](https://openaccess.city.ac.uk/id/eprint/25810/1/ESWC2021_ontology_alignment_LogMap_ML.pdf)

This repository includes the implementation of LogMap-ML introduced in the paper ****Augmenting Ontology Alignment by Semantic Embedding and Distant Supervision**** (see also [LogMap](https://github.com/ernestojimenezruiz/logmap-matcher/)).

The HeLis and FoodOn ontologies, and their partial GS, which are adopted for the evaluation in the paper, are under `data/`.
Note the HeLis ontology adopted has been pre-processed by transforming instances into classes.

## Install 

Run:

```sh
$ python3 -m venv env

$ source env/bin/activate

$ python -m pip install -r requirements.txt
```

## Usage

### Pre-process #1: Run the original system

Run LogMap:

```sh
$ java -jar logmap/logmap-matcher-4.0.jar MATCHER file:/path/to/data/helis_v1.00.owl file:/path/to/data/foodon-merged.owl /path/to/data/logmap_output/ true
```

This leads to LogMap anchor mappings
(<img src="https://render.githubusercontent.com/render/math?math=\mathcal{M}_a">)
and
over-estimation class mappings
(<img src="https://render.githubusercontent.com/render/math?math=\mathcal{M}_o">).

### Pre-process #2: Get Embedding Models

You can either download the word2vec embedding by gensim (the one trained by English Wikipedia articles in 2018-[download](https://drive.google.com/file/d/1rm9uJEKG25PJ79zxbZUWuaUroWeoWbFR/view?usp=sharing)):

```sh
$ python -m gensim.downloader --download fasttext-wiki-news-subwords-300
```

or use the ontology-tailored [OWL2Vec\*](https://github.com/KRR-Oxford/OWL2Vec-Star) embedding. 

The ontologies can use their own embedding models or use one common embedding model.

### Pre-process #3: Class Name and Path Extraction

This is to extract the class name and path information for each class in an ontology:

```sh
$ python scripts/extraction.py data/helis_v1.00.owl
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
$ python scripts/sample.py data/logmap_output/logmap_anchors.txt --left_names data/helis_v1.00_names.json --left_paths data/helis_v1.00_paths.txt --right_names data/foodon-merged_names.json --right_paths data/foodon-merged_paths.txt
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

### Step #2: Train, Valid and Predict - TODO

We train a SiamNN as the mapping prediction model:

```sh
$ python scripts/train_valid.py --left_w2v_dir w2v/ --left_w2v_dir w2v/ --train_mappings data/train_mappings.txt --valid_mappings data/validation_mappings.txt --nn_dir data/model
```

```sh
$ python scripts/predict_candidates.py data/logmap_output/logmap_overestimation.txt --left_names data/helis_v1.00_names.json --left_paths data/helis_v1.00_paths.txt --right_names data/foodon-merged_names.json --right_paths data/foodon-merged_paths.txt --left_w2v_dir w2v/ --left_w2v_dir w2v/```
```

### Step #3: Evaluate - TODO

#### With approximation

Calculate the recall w.r.t. the GS, and sample a number of mappings for annotation, by:

```python evaluate_for_approximate.py --threshold 0.5 --anchor_file logmap_output/logmap_anchors.txt```

It will output a file with a part of the mappings for human annotation. 
The annotation is done by appending "true" or "false" to each mapping (see annotation example in evaluate.py).
With the manual annotation and the GS, the precision and recall can be approximated by:

```python approximate_precision_recall.py```

Please see Eq. (2)(3)(4) in the paper for how the precision and recall approximation works.
For more accurate approximate, it is suggested to annotate and use the mappings of at least three systems to approximate the GS. 
Besides the original LogMap and LogMap-ML, you can also consider [AML](https://github.com/AgreementMakerLight/AML-Project) as well.

#### Straightforward 
If it is assumed that gold standards (complete ground truth mappings) are given, Precision and Recall can be directly calculated by:

```python evaluate_straightforward.py --prediction_out_file data/predict_score.txt --oaei_GS_file reference.rdf```

## Publications

* Jiaoyan Chen, Ernesto Jimenez-Ruiz, Ian Horrocks, Denvar Antonyrajah, Ali Hadian, Jaehun Lee. **Augmenting Ontology Alignment by Semantic Embedding and Distant Supervision**. European Semantic Web Conference, ESWC 2021. ([PDF](https://openaccess.city.ac.uk/id/eprint/25810/1/ESWC2021_ontology_alignment_LogMap_ML.pdf))
