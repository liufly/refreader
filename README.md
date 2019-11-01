RefReader
========

This repository hosts the implementation of [the referential reader](https://www.aclweb.org/anthology/P19-1593), built on a fork of [fairseq](https://github.com/pytorch/fairseq).

```
@inproceedings{liu-etal-2019-referential,
    title     = "The Referential Reader: A Recurrent Entity Network for Anaphora Resolution",
    author    = "Liu, Fei and Zettlemoyer, Luke and Eisenstein, Jacob",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month     = jul,
    year      = "2019",
    address   = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url       = "https://www.aclweb.org/anthology/P19-1593",
    doi       = "10.18653/v1/P19-1593",
    pages     = "5918--5925",
}
```

# Environment

```
Python 3.6
PyTorch 1.1
tqdm 4.32.1
```

# Prepare GAP text for BERT pre-processing
Replace `${REFREADER_PATH}` with path to your local copy of the RefReader repo.
```
$ tail -n +2 ${REFREADER_PATH}/data/gap/gap-development.tsv | cut -d$'\t' -f2 > ${REFREADER_PATH}/data/gap/gap-development.txt
$ tail -n +2 ${REFREADER_PATH}/data/gap/gap-test.tsv | cut -d$'\t' -f2 > ${REFREADER_PATH}/data/gap/gap-test.txt
$ tail -n +2 ${REFREADER_PATH}/data/gap/gap-validation.tsv | cut -d$'\t' -f2 > ${REFREADER_PATH}/data/gap/gap-validation.txt
```

# Prepare GAP text for fairseq pre-processing
Extract text from (BERT-)tokenized GAP tsv files (credit goes to [Mandar Joshi](https://homes.cs.washington.edu/~mandar90/) for preparing these files):
```
$ tail -n +2 ${REFREADER_PATH}/data/gap/gap-development.bert.tsv | cut -d$'\t' -f2 > ${REFREADER_PATH}/data/gap/gap-development.bert.txt
$ tail -n +2 ${REFREADER_PATH}/data/gap/gap-test.bert.tsv | cut -d$'\t' -f2 > ${REFREADER_PATH}/data/gap/gap-test.bert.txt
$ tail -n +2 ${REFREADER_PATH}/data/gap/gap-validation.bert.tsv | cut -d$'\t' -f2 > ${REFREADER_PATH}/data/gap/gap-validation.bert.txt
```
Construct dictionary:
```
$ python txt2dict.py ${REFREADER_PATH}/data/gap/gap-development.bert.txt ${REFREADER_PATH}/data/gap/gap-test.bert.txt ${REFREADER_PATH}/data/gap/gap-validation.bert.txt ${REFREADER_PATH}/data/gap/gap-bert.dict
```

# Fairseq pre-processing
```
$ python preprocess.py --only-source --trainpref ${REFREADER_PATH}/data/gap/gap-development.bert.txt --validpref ${REFREADER_PATH}/data/gap/gap-validation.bert.txt --testpref ${REFREADER_PATH}/data/gap/gap-test.bert.txt --destdir ${REFREADER_PATH}/data/gap-bert-bin/ --srcdict ${REFREADER_PATH}/data/gap/gap-bert.dict

```
And then copy and re-name files:
```
$ cp ${REFREADER_PATH}/data/gap/gap-development.bert.tsv ${REFREADER_PATH}/data/gap-bert-bin/gap-train.bert.tsv
$ cp ${REFREADER_PATH}/data/gap/gap-validation.bert.tsv ${REFREADER_PATH}/data/gap-bert-bin/gap-valid.bert.tsv
$ cp ${REFREADER_PATH}/data/gap/gap-test.bert.tsv ${REFREADER_PATH}/data/gap-bert-bin/gap-test.bert.tsv
```

# Extract BERT features for GAP text
+ Clone code at [BERT GitHub repo](https://github.com/google-research/bert) and follow the instructions in [README.md](https://github.com/google-research/bert/blob/master/README.md) to configure the environment.
+ Download [BERT pre-trained model](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip).
+ Replace `${BERT_PATH}` with path to your local copy of the BERT repo.


- Extract BERT features for `gap-development.txt` `gap-train.bert.jsonl`
```
$ python ${BERT_PATH}/extract_features.py --input_file=${REFREADER_PATH}/data/gap/gap-development.txt --output_file=${REFREADER_PATH}/data/gap-bert-bin/gap-train.bert.jsonl --vocab_file=${BERT_MODEL}/cased_L-12_H-768_A-12/vocab.txt --bert_config_file=${BERT_MODEL}/cased_L-12_H-768_A-12/bert_config.json --init_checkpoint=${BERT_MODEL}/cased_L-12_H-768_A-12/bert_model.ckpt --layers=-1,-2,-3,-4 --max_seq_length=512 --batch_size=1 --do_lower_case=False
```
- Extract BERT features for `gap-validation.txt` `gap-valid.bert.jsonl`
```
$ python ${BERT_PATH}/extract_features.py --input_file=${REFREADER_PATH}/data/gap/gap-validation.txt --output_file=${REFREADER_PATH}/data/gap-bert-bin/gap-valid.bert.jsonl --vocab_file=${BERT_MODEL}/cased_L-12_H-768_A-12/vocab.txt --bert_config_file=${BERT_MODEL}/cased_L-12_H-768_A-12/bert_config.json --init_checkpoint=${BERT_MODEL}/cased_L-12_H-768_A-12/bert_model.ckpt --layers=-1,-2,-3,-4 --max_seq_length=512 --batch_size=1 --do_lower_case=False
```
- Extract BERT features for `gap-test.txt` `gap-test.bert.jsonl`
```
$ python ${BERT_PATH}/extract_features.py --input_file=${REFREADER_PATH}/data/gap/gap-test.txt --output_file=${REFREADER_PATH}/data/gap-bert-bin/gap-test.bert.jsonl --vocab_file=${BERT_MODEL}/cased_L-12_H-768_A-12/vocab.txt --bert_config_file=${BERT_MODEL}/cased_L-12_H-768_A-12/bert_config.json --init_checkpoint=${BERT_MODEL}/cased_L-12_H-768_A-12/bert_model.ckpt --layers=-1,-2,-3,-4 --max_seq_length=512 --batch_size=1 --do_lower_case=False
```

# Training
```
$ python train.py --task language_modeling_refreader_gap_bert --arch refreader_gap_bert data/gap-bert-bin/ --restore-file=NoFile --distributed-world-size=1 --max-tokens 5000 --lr 1e-3 --optimizer adam --train-subset train --valid-subset valid --criterion gatekeeper_gap_bert --no-epoch-checkpoints --save-dir ${SAVE_DIR}
```

When training ends, the final output should look like this:
```
| epoch 035 | valid on 'valid' subset | valid_loss -78.4106 | valid_ppl 24505.21 | num_updates 2975 | best 80 | best_threshold 0.07 | f@0.04 78.4106 | mf@0.04 78.5908 | ff@0.04 78.2383
```
+ the last three items show the F-scores based on the current model in three categories at the threshold value of `@0.04` on the `valid` set:
    + overall: `f@0.04 78.4106`
    + masculine `mf@0.04 78.5908`
    + feminine `ff@0.04 78.2383`
+ `best 80` indicates the best overall F-score up to the current epoch on the `valid` set (the snapshot of this best performing model is saved at `${SAVE_DIR}`)
+ `best_threshold 0.07` is the threshold value the `best 80` validation F-score was achieved at

# Predict
```
$ python train.py --task language_modeling_refreader_gap_bert --arch refreader_gap_bert data/gap-bert-bin/ --distributed-world-size=1 --max-tokens 5000 --lr 1e-3 --optimizer adam --train-subset train --valid-subset test --criterion gatekeeper_gap_bert --no-epoch-checkpoints --restore-dir ${SAVE_DIR} --restore-file checkpoint_best.pt --threshold ${BEST_THRESHOLD} --no-save --no-train
```
