wget https://github.com/mmihaltz/word2vec-GoogleNews-vectors/blob/master/GoogleNews-vectors-negative300.bin.gz -P data/
wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip -P data/
apt-get install unzip
unzip ./data/snli_1.0.zip -d ./data/
mv ./data/snli_1.0/snli_1.0_dev.jsonl ./data/snli_1.0_dev.json
mv ./data/snli_1.0/snli_1.0_train.jsonl ./data/snli_1.0_train.json
mv ./data/snli_1.0/snli_1.0_test.jsonl ./data/snli_1.0_test.json
cd data
gunzip GoogleNews-vectors-negative300.bin.gz
cd ..
