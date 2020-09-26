import os
import json
from pytools import memoize_method

from onir import datasets, util, indices
from onir.interfaces import trec, plaintext


@datasets.register('qulac')
class QulacDataset(datasets.IndexBackedDataset):
    """
    Interface to the Qulac dataset.
    Introduced by Aliannejadi et. al. Asking Clarifying Questions in Open-Domain Information-Seeking Conversations
    """

    def __init__(self, config, logger, vocab):
        super().__init__(config, logger, vocab)
        base_path = "../src/data/qulac"
        self.base_path = "../src/data/qulac"
        self.index = indices.AnseriniIndex(os.path.join(base_path, 'anserini'), stemmer='none')
        self.index_stem = indices.AnseriniIndex(os.path.join(base_path, 'anserini.porter'), stemmer='porter')
        self.doc_store = indices.SqliteDocstore(os.path.join(base_path, 'docs.sqlite'))

    def init(self, force=False):
        pass

    def qrels(self, fmt='dict'):
        return self._load_qrels(self.config['subset'], fmt)

    @memoize_method
    def _load_qrels(self, subset, fmt):
        # TODO: change dummy to real file after parsing is implemented
        return trec.read_qrels_fmt(os.path.join(self.base_path, f'{subset}.qrels.dummy.txt'), fmt)

    def _get_index(self, record):
        return self.index

    def _get_docstore(self):
        return self.doc_store

    def _get_index_for_batchsearch(self):
        return self.index_stem

    @memoize_method
    def _load_queries_base(self, subset):
        result = {}

        qulac = self._load_qulac_data()
        query_ids = qulac['topic_id']
        queries = qulac['topic']

        for i in range(len(query_ids)):
            qid = query_ids[str(i)]
            query_text = queries[str(i)]
            if qid not in result:
                result[qid] = query_text

        return result

    @memoize_method
    def _load_queries_QQA(self, subset):
        result = {}

        qulac = self._load_qulac_data()
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
        f = os.path.join(self.base_path, f'{subset}.queries.txt')
        for qid, text in plaintext.read_tsv(f):
            result[qid] = text
        return result

    def _load_qulac_data(self):
        # TODO split qulac in train and test
        with open(os.path.join(self.base_path, 'qulac.json')) as f:
            qulac = json.load(f)
        return qulac


def create_qrel_file():
    """
    TODO: Looks like we have to parse 'qulac.json' and 'qulac_hist012_dict.json' to qrel format.
          => https://trec.nist.gov/data/qrels_eng/
    """
    pass


def create_queries_files():
    pass
