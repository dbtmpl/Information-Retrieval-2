import os
import json
from pytools import memoize_method

import numpy as np

from onir import datasets, util, indices
from onir.interfaces import trec, plaintext

from utils.data_utils import load_json


@datasets.register('qulac')
class QulacDataset(datasets.IndexBackedDataset):
    """
    Interface to the Qulac dataset.
    Introduced by Aliannejadi et. al. Asking Clarifying Questions in Open-Domain Information-Seeking Conversations
    """
    DUA = """
        At this point permission for downloading the dataset is asked. Does Not apply in our case. Answer 'yes'. 
        """

    def __init__(self, config, logger, vocab):
        super().__init__(config, logger, vocab)
        self.qulac_base = "../src/data/qulac"
        self.doc_base = "../src/data/documents/webclue_docs"
        self.index = indices.AnseriniIndex(os.path.join(self.qulac_base, 'anserini'), stemmer='none')
        self.index_stem = indices.AnseriniIndex(os.path.join(self.qulac_base, 'anserini.porter'), stemmer='porter')
        self.doc_store = indices.SqliteDocstore(os.path.join(self.qulac_base, 'docs.sqlite'))

    @staticmethod
    def default_config():
        result = datasets.IndexBackedDataset.default_config()
        result.update({
            'subset': 'train'
        })
        return result

    def init(self, force=False):
        """
        No need to download data. Thus only indexing is happening.
        """

        idxs = [self.index, self.index_stem, self.doc_store]
        self._init_indices_parallel(idxs, self._init_iter_collection(), force)

    def qrels(self, fmt='dict'):
        return self._load_qrels(self.config['subset'], fmt)

    def _init_iter_collection(self):
        # Load first 1 doc as a test
        for i in range(1):
            self.logger.info(f'loading {i + 1}.json ...')
            doc_i = load_json(os.path.join(self.doc_base, f'{i + 1}.json'))
            doc_ids = doc_i['id']
            doc_texts = doc_i['text']

            for j in range(len(doc_ids)):
                did = doc_ids[str(j)]
                doc_text = doc_texts[str(j)]
                yield indices.RawDoc(did, doc_text)

    @memoize_method
    def _load_qrels(self, subset, fmt):
        # TODO: change dummy to real file after parsing is implemented
        return trec.read_qrels_fmt(os.path.join(self.qulac_base, f'{subset}.qrels.txt'), fmt)

    def _get_index(self, record):
        return self.index

    def _get_docstore(self):
        return self.doc_store

    def _get_index_for_batchsearch(self):
        return self.index_stem

    @memoize_method
    def _load_queries_base(self, subset):
        result = {}

        qulac = load_json(os.path.join(self.qulac_base, 'qulac.json'))
        query_ids = qulac['topic_id']
        queries = qulac['topic']

        for i in range(len(query_ids)):
            qid = query_ids[str(i)]
            query_text = queries[str(i)]
            if qid not in result:
                result[str(qid)] = query_text

        print(result)
        return result

    @memoize_method
    def _load_queries_QQA(self, subset):
        result = {}

        qulac = load_json(os.path.join(self.qulac_base, 'qulac.json'))
        query_ids = qulac['topic_id']
        queries = qulac['topic']

        # TODO: Integrate Q_0, Q and A here in a sane manner. Depends on remaining architecture.
        for i in range(len(query_ids)):
            qid = query_ids[i]
            query_text = queries[i]
            if qid not in result:
                result[qid] = query_text

        return result

    @memoize_method
    def _load_queries_multi_turn(self, subset):
        result = {}
        f = os.path.join(self.qulac_base, f'{subset}.queries.txt')
        for qid, text in plaintext.read_tsv(f):
            result[qid] = text
        return result

    def _load_qulac_data(self):
        # TODO split qulac in train and test
        with open(os.path.join(self.qulac_base, 'qulac.json')) as f:
            qulac = json.load(f)
        return qulac


def create_dummy_qrel_file():
    """
    TODO: Looks like we have to parse 'qulac.json' and 'qulac_hist012_dict.json' to qrel format.
          => https://trec.nist.gov/data/qrels_eng/
    """
    doc_base = "../data/documents/webclue_docs"
    i = 1
    doc_i = load_json(os.path.join(doc_base, f'{i}.json'))
    doc_ids = doc_i['id']
    num_docs = len(doc_ids)
    qid = np.random.randint(200, size=num_docs)
    rels = [-2, 0, 1, 2, 3, 4]
    rel_inds = np.random.choice(len(rels), num_docs)

    with open('../data/qulac/output.txt', 'a') as f1:
        for i in range(len(doc_ids)):
            line = f"{qid[i]}  0  {doc_ids[str(i)]}  {rels[rel_inds[i]]}"
            f1.write(line + "\n")


def create_queries_files():
    pass


if __name__ == "__main__":
    create_dummy_qrel_file()
