import json
import os
import pickle

import numpy as np


def load_json(path):
    with open(path) as f:
        qulac = json.load(f)
    return qulac


def split_data():
    """
    Function splitting data into train, valid, test
    """

    with open('../data/qulac/qids_splits_topic.pkl', 'rb') as f:
        _splits = pickle.load(f)

    train_ids, val_ids, test_ids = _splits['train'], _splits['val'], _splits['test']

    with open("../data/qulac/faceted_qid.qrel", "r") as qrels:
        for qrel in qrels:
            qrel_line = qrel.strip()
            qrel_line_split = qrel_line.split()
            topic, _, doc_id, relevancy = qrel_line_split

            if topic in train_ids:
                with open(f'../data/qulac/train.qrels.txt', 'a') as train_file:
                    train_file.write(qrel_line + "\n")
            elif topic in val_ids:
                with open(f'../data/qulac/valid.qrels.txt', 'a') as train_file:
                    train_file.write(qrel_line + "\n")
            elif topic in test_ids:
                with open(f'../data/qulac/test.qrels.txt', 'a') as train_file:
                    train_file.write(qrel_line + "\n")
            else:
                print(f"Unknown topic id: {topic}! Not added to any file.")


def create_dummy_qrel_file():
    """
    Ugly but simple way to create a dummy qrel file.
    """
    doc_base = "../data/documents/webclue_docs"
    i = 1
    doc_i = load_json(os.path.join(doc_base, f'{i}.json'))
    doc_ids = doc_i['id']
    num_docs = len(doc_ids)
    qid = np.random.randint(200, size=num_docs)
    rels = [-2, 0, 1, 2, 3, 4]
    rel_inds = np.random.choice(len(rels), num_docs)

    with open('../data/qulac/train.qrels.txt', 'a') as f1:
        for i in range(len(doc_ids)):
            line = f"{qid[i]}  0  {doc_ids[str(i)]}  {rels[rel_inds[i]]}"
            f1.write(line + "\n")


if __name__ == "__main__":
    # create_dummy_qrel_file()
    split_data()
