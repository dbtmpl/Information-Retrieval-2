import json
import os
import pickle

import pandas as pd
import numpy as np


def load_json(path):
    with open(path) as f:
        dict = json.load(f)
    return dict


def save_json(path, dict):
    with open(path, 'w') as f:
        json.dump(dict, f)


def save_pickle(name, thing):
    with open(name, 'wb') as handle:
        pickle.dump(thing, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(name):
    with open(name, 'rb') as handle:
        thing = pickle.load(handle)
    return thing


def save_docs_ids_we_use():
    doc_names = os.listdir("../data/documents/webclue_docs_1000")

    used_doc_ids = []
    for i, doc_name in enumerate(doc_names):
        print(f'processing {doc_name} ... [{i + 1} / {len(doc_names)}]')
        docs_i = load_json(os.path.join("../data/documents/webclue_docs_1000", doc_name))
        doc_ids = list(docs_i['id'].values())
        used_doc_ids.append(doc_ids)

    save_pickle('../data/qulac/used_docs.pkl', used_doc_ids)


def keep_top1000_docs_per_topic(path_all_docs, path_1000k_docs):
    doc_names = os.listdir(path_all_docs)
    docs_sorted = load_pickle('../data/qulac/initial_retrieval_dict_10K.pkl')

    for i, doc_name in enumerate(doc_names):
        print(f'processing {doc_name} ... [{i + 1} / {len(doc_names)}]')
        top_1000_docs = {
            'id': {},
            'text': {}
        }
        topic_id = doc_name.split('.')[0]
        docs_i = load_json(os.path.join(path_all_docs, doc_name))
        keep_doc_ids_i = docs_sorted[int(topic_id)]

        for j, doc_id_sorted in enumerate(keep_doc_ids_i[:1000]):
            for idx, doc_id_json in docs_i['id'].items():

                if doc_id_sorted == doc_id_json:
                    top_1000_docs['id'][j] = doc_id_sorted
                    top_1000_docs['text'][j] = docs_i['text'][idx]

        save_json(os.path.join(path_1000k_docs, f"{topic_id}.json"), top_1000_docs)


def split_data():
    """
    Function splitting data into train, valid, test
    """

    _splits = load_pickle('../data/qulac/qids_splits_topic.pkl')
    _ud = load_pickle('../data/qulac/used_docs.pkl')

    _used_docs = [item for sublist in _ud for item in sublist]

    train_ids, val_ids, test_ids = _splits['train'], _splits['val'], _splits['test']

    train_size, val_size, test_size = 0, 0, 0

    with open("../data/qulac/faceted_qid.qrel", "r") as qrels:
        for qrel in qrels:
            qrel_line = qrel.strip()
            qrel_line_split = qrel_line.split()
            topic, _, doc_id, relevancy = qrel_line_split

            if doc_id.strip() not in _used_docs:
                continue

            print(f"Current split sizes: train_size: {train_size}, val_size:{val_size}, test_size:{test_size}")

            if topic in train_ids:
                train_size += 1
                with open(f'../data/qulac/train.qrels.txt', 'a') as train_file:
                    train_file.write(qrel_line + "\n")
            elif topic in val_ids:
                val_size += 1
                with open(f'../data/qulac/valid.qrels.txt', 'a') as train_file:
                    train_file.write(qrel_line + "\n")
            elif topic in test_ids:
                test_size += 1
                with open(f'../data/qulac/test.qrels.txt', 'a') as train_file:
                    train_file.write(qrel_line + "\n")
            else:
                print(f"Unknown topic id: {topic}! Not added to any file.")

    print(f"Data split sizes: train_size: {train_size}, val_size:{val_size}, test_size:{test_size}")


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


def gen_split_test_qrel(full_set_path, destination):
    qulac = load_json('../data/qulac/qulac.json')
    qulac_df = pd.DataFrame.from_dict(qulac)
    qulac_df['ans_tokens'] = qulac_df.answer.apply(lambda x: x.split(' '))

    neg_tokens = set(('none', 'no', 'neither', 'nope'))  # maybe also not?
    pos_tokens = set(('yes',))
    qulac_df['Ans_yes'] = qulac_df.ans_tokens.apply(lambda x: len(pos_tokens & set(x)) > 0)
    qulac_df['Ans_no'] = qulac_df.ans_tokens.apply(lambda x: (len(neg_tokens & set(x)) > 0) or
                                                             (len({'not', 'related'} & set(x)) == 2)
                                                   )
    qulac_df['Ans_idk'] = qulac_df.ans_tokens.apply(lambda x: (len({'dont', 'know'} & set(x)) == 2) or
                                                              (len({'not', 'sure'} & set(x)) == 2)
                                                    )
    qulac_df['Ans_other'] = ~(qulac_df.Ans_yes | qulac_df.Ans_no | qulac_df.Ans_idk)
    qulac_df['polarity'] = qulac_df.apply(lambda x: '+' if x.Ans_yes else
    ('-' if x.Ans_no else
     ('idk' if x.Ans_idk else 'N'
      )), axis=1)

    yes_tfq_ids = qulac_df[qulac_df['Ans_yes'] == True].topic_facet_question_id.tolist()
    no_tfq_ids = qulac_df[qulac_df['Ans_no'] == True].topic_facet_question_id.tolist()
    idk_tfq_ids = qulac_df[qulac_df['Ans_idk'] == True].topic_facet_question_id.tolist()
    other_tfq_ids = qulac_df[qulac_df['Ans_other'] == True].topic_facet_question_id.tolist()

    test_set = pd.read_csv(full_set_path, delimiter='\s+', index_col=False, header=None, names=['tfq', 'a', 'b', 'c'])

    test_set[test_set['tfq'].isin(yes_tfq_ids)].to_csv(destination + 'yes_test.qrel.txt', header=None, index=None,
                                                       sep=' ', mode='a')
    test_set[test_set['tfq'].isin(no_tfq_ids)].to_csv(destination + 'no_test.qrel.txt', header=None, index=None,
                                                      sep=' ', mode='a')
    test_set[test_set['tfq'].isin(idk_tfq_ids)].to_csv(destination + 'idk_test.qrel.txt', header=None, index=None,
                                                       sep=' ', mode='a')
    test_set[test_set['tfq'].isin(other_tfq_ids)].to_csv(destination + 'other_test.qrel.txt', header=None, index=None,
                                                         sep=' ', mode='a')


if __name__ == "__main__":
    gen_split_test_qrel('../data/qulac/test.qrels.txt', '../data/qulac/')
    # keep_top1000_docs_per_topic("../data/documents/webclue_docs", "../data/documents/webclue_docs_1000/")
    # save_docs_ids_we_use()
    # split_data()

