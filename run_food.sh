python extraction.py use_cases/food/helis_v1.00.owl --prefix left_

python extraction.py use_cases/food/foodon-merged.owl --prefix right_

python sample.py logmap_output_food/logmap_anchors.txt \
  --left-names left_names.json --right-names right_names.json

## a

python train_valid.py --train-mappings train_mappings.txt --valid-mappings validation_mappings.txt \
  --left-owl2vec owl2vec_model/owl2vec_food/output --right-owl2vec owl2vec_model/owl2vec_food/output \
  --left-w2v enwiki_model/word2vec_gensim --right-w2v enwiki_model/word2vec_gensim \
  --encoder-type word-con --nn-dir model

python predict_candidates.py logmap_output_food/logmap_overestimation.txt \
  --left-names left_names.json --right-names right_names.json \
  --left-owl2vec owl2vec_model/owl2vec_food/output --right-owl2vec owl2vec_model/owl2vec_food/output \
  --left-w2v enwiki_model/word2vec_gensim --right-w2v enwiki_model/word2vec_gensim \
  --encoder-type word-con --nn-dir model

python evaluate.py --GS use_cases/food/reference.rdf \
  --anchors logmap_output_food/logmap_anchors.txt --prediction prediction.txt --threshold 0.9

# b

python train_valid.py --train-mappings train_mappings.txt --valid-mappings validation_mappings.txt \
  --left-owl2vec owl2vec_model/owl2vec_food/output --right-owl2vec owl2vec_model/owl2vec_food/output \
  --left-w2v enwiki_model/word2vec_gensim --right-w2v enwiki_model/word2vec_gensim \
  --encoder-type word-con --nn-dir model

python predict_candidates.py logmap_output_food/logmap_anchors.txt \
  --left-names left_names.json --right-names right_names.json \
  --left-owl2vec owl2vec_model/owl2vec_food/output --right-owl2vec owl2vec_model/owl2vec_food/output \
  --left-w2v enwiki_model/word2vec_gensim --right-w2v enwiki_model/word2vec_gensim \
  --encoder-type word-con --nn-dir model

python evaluate.py --GS logmap_output_food/logmap_anchors.rdf \
  --prediction prediction.txt --threshold 0.9
