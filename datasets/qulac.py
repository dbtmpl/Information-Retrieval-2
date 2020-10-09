import os
import contextlib
import functools

from pytools import memoize_method

from onir import datasets, indices, util
from onir.interfaces import trec
from onir.datasets.index_backed import LazyDataRecord

from utils.data_utils import load_json, load_pickle


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
        self.doc_base = "../src/data/documents/webclue_docs_1000"
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

    def _init_indices_parallel(self, indices, doc_iter, force):
        needs_docs = []
        for index in indices:
            if force or not index.built():
                needs_docs.append(index)

        if needs_docs:
            with contextlib.ExitStack() as stack:
                doc_iters = util.blocking_tee(doc_iter, len(needs_docs))
                for idx, it in zip(needs_docs, doc_iters):
                    stack.enter_context(util.CtxtThread(functools.partial(idx.build, it)))

    def build_record(self, fields, **initial_values):
        record = LazyDataRecordQulac(self, **initial_values)
        record.load(fields)
        return record

    def qrels(self, fmt='dict'):
        return self._load_qrels(self.config['subset'], fmt)

    def _init_iter_collection(self):
        doc_names = os.listdir(self.doc_base)

        for i, doc_name in enumerate(doc_names):
            self.logger.info(f'loading {doc_name} ... [{i + 1} / {len(doc_names)}]')
            docs_i = load_json(os.path.join(self.doc_base, doc_name))
            doc_ids = docs_i['id']
            doc_texts = docs_i['text']

            for j in range(len(doc_ids)):
                did = doc_ids[str(j)]
                doc_text = doc_texts[str(j)]
                yield indices.RawDoc(did, doc_text)

    @memoize_method
    def _load_qrels(self, subset, fmt):
        return trec.read_qrels_fmt(os.path.join(self.qulac_base, f'{subset}.qrels.txt'), fmt)

    def _get_index(self, record):
        return self.index

    def _get_docstore(self):
        return self.doc_store

    def _get_index_for_batchsearch(self):
        return self.index_stem

    def _question_rawtext(self, record):
        return self._load_questions_base(self.config['subset'])[record['query_id']]

    def _question_text(self, record):
        return tuple(self.vocab.tokenize(record['question_rawtext']))

    def _question_tok(self, record):
        return [self.vocab.tok2id(t) for t in record['question_text']]

    def _question_idf(self, record):
        index = self._get_index(record)
        return [index.term2idf(t) for t in record['question_text']]

    def _question_len(self, record):
        return len(record['question_text'])

    def _answer_rawtext(self, record):
        return self._load_answers_base(self.config['subset'])[record['query_id']]

    def _answer_text(self, record):
        return tuple(self.vocab.tokenize(record['answer_rawtext']))

    def _answer_tok(self, record):
        return [self.vocab.tok2id(t) for t in record['answer_text']]

    def _answer_idf(self, record):
        index = self._get_index(record)
        return [index.term2idf(t) for t in record['answer_text']]

    def _answer_len(self, record):
        return len(record['answer_text'])

    @memoize_method
    def _load_queries_base(self, subset):
        result = {}

        qulac = load_json(os.path.join(self.qulac_base, 'qulac.json'))
        query_ids = qulac['topic_facet_question_id']
        queries = qulac['topic']

        for i in range(len(query_ids)):
            qid = query_ids[str(i)]
            # if qid not in _split:
            #     continue
            query_text = queries[str(i)].strip()
            result[str(qid)] = query_text

        return result

    @memoize_method
    def _load_questions_base(self, subset):
        result = {}

        qulac = load_json(os.path.join(self.qulac_base, 'qulac.json'))
        query_ids = qulac['topic_facet_question_id']
        questions = qulac['question']

        for i in range(len(query_ids)):
            qid = query_ids[str(i)]
            # if qid not in _split:
            #     continue
            question = questions[str(i)].strip()
            result[str(qid)] = question

        return result

    @memoize_method
    def _load_answers_base(self, subset):
        result = {}

        qulac = load_json(os.path.join(self.qulac_base, 'qulac.json'))
        query_ids = qulac['topic_facet_question_id']
        answers = qulac['answer']

        for i in range(len(query_ids)):
            qid = query_ids[str(i)]
            # if qid not in _split:
            #     continue
            answer = answers[str(i)].strip()
            result[str(qid)] = answer

        return result


class LazyDataRecordQulac(LazyDataRecord):
    def __init__(self, ds, **data):
        # pylint: disable=W0212
        super().__init__(ds, **data)
        self.ds = ds
        self._data = data
        self.methods = {
            'answer_rawtext': ds._answer_rawtext,
            'answer_text': ds._answer_text,
            'answer_tok': ds._answer_tok,
            'answer_idf': ds._answer_idf,
            'answer_len': ds._answer_len,
            'question_rawtext': ds._question_rawtext,
            'question_text': ds._question_text,
            'question_tok': ds._question_tok,
            'question_idf': ds._question_idf,
            'question_len': ds._question_len,
            'query_rawtext': ds._query_rawtext,
            'query_text': ds._query_text,
            'query_tok': ds._query_tok,
            'query_idf': ds._query_idf,
            'query_len': ds._query_len,
            'query_lang': ds._query_lang,
            'query_score': ds._query_score,
            'doc_rawtext': ds._doc_rawtext,
            'doc_text': ds._doc_text,
            'doc_tok': ds._doc_tok,
            'doc_idf': ds._doc_idf,
            'doc_len': ds._doc_len,
            'doc_lang': ds._doc_lang,
            'runscore': ds._runscore,
            'relscore': ds._relscore,
            'kdescore': ds._kdescore,
            'normscore': ds._normscore,
            'rank': ds._rank,
        }
