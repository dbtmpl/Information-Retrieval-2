rm -rf ../src/data/qulac/anserini.porter.test.runs ../src/data/qulac/docs.sqlite
rm -rf ~/data/onir/models/default/conv_knrm_qqa_30q_475d_ng3_conv128/wordvec_hash_qqa_fasttext_wiki-news-300d-1M/pairwise_32x16_adam-0.001_softmax_pos-intersect-query-minrel3_neg-run/qulac_train_bm25_k1-1.4_b-0.40.100/qulac_test_bm25_k1-1.4_b-0.40.50

cat ../src/data/qulac/yes_test_qrel.txt > ../src/data/qulac/test.qrels.txt

python ../src/start.py ../src/config/ranker/std_neural_rankers/conv_knrm_qqa/ ../src/config/qulac