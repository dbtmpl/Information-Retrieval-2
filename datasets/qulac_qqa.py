from pytools import memoize_method
import os
import json
import numpy as np

from onir import datasets, util, indices
from datasets.qulac import QulacDataset
from onir.interfaces import trec, plaintext
from utils.data_utils import load_json

# TODO add correct mappings from ids to documents

@datasets.register('qulac_qqa')
class QulacQQADataset(QulacDataset):
    def __init__(self, config, logger, vocab):
        super(QulacDataset, self).__init__(config, logger, vocab)
        self.qulac_base = "../src/data/qulac"
        self.doc_base = "../src/data/documents/webclue_docs"
        self.index = indices.AnseriniIndex(os.path.join(self.qulac_base, 'anserini'), stemmer='none')
        self.index_stem = indices.AnseriniIndex(os.path.join(self.qulac_base, 'anserini.porter'), stemmer='porter')
        self.doc_store = indices.SqliteDocstore(os.path.join(self.qulac_base, 'docs.sqlite'))

    @memoize_method
    def _load_queries_base(self, subset):
        """
        Should be similar to _load_queries_base but with integrating Q_0, Q and A.
        I guess we need to write a new 'pair_iter' function for this and adapt the Trainer.
        """

        seperator = '<SEP>'

        result = {}
        qulac = load_json(os.path.join(self.qulac_base, 'qulac.json'))
        topic_ids = qulac['topic_id']
        queries = qulac['topic']

        for i in range(len(qulac['topic_id'])):
            qid = qulac['topic_id'][str(i)]
            
            query = qulac['topic'][str(i)]
            question = qulac['question'][str(i)]
            answer = qulac['answer'][str(i)]

            query_text = query + ' ' + seperator + ' ' + question + ' ' + seperator + ' ' + answer
            result[str(i)] = query_text
        return result