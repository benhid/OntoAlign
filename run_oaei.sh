python extraction.py use_cases/oaei_2021/oaei.owl --prefix conference_

python sample.py logmap_output_oaei_2021/logmap_anchors.txt \
  --left-names conference_names.json --right-names conference_names.json

python train_valid.py --train-mappings train_mappings.txt --valid-mappings validation_mappings.txt \
  --left-owl2vec owl2vec_model/owl2vec_food/output --right-owl2vec owl2vec_model/owl2vec_food/output \
  --left-w2v enwiki_model/word2vec_gensim --right-w2v enwiki_model/word2vec_gensim \
  --nn-dir model

python predict_candidates.py logmap_output/logmap_overestimation.txt \
  --left-names left_names.json --right-names right_names.json \
  --left-owl2vec owl2vec_model/owl2vec_food/output --right-owl2vec owl2vec_model/owl2vec_food/output \
  --left-w2v enwiki_model/word2vec_gensim --right-w2v enwiki_model/word2vec_gensim \
  --encoder-type word-con+vector --nn-dir model

python evaluate.py --GS use_cases/food/reference.rdf \
  --anchors logmap_output/logmap_anchors.txt --prediction prediction.txt --threshold 0.9

python evaluate.py --GS logmap_output/logmap_anchors.rdf \
  --prediction prediction.txt --threshold 0.9
