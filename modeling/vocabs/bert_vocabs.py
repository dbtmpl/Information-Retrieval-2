import torch

from onir.vocab import BertVocab
from onir.vocab.bert_vocab import SepBertEncoder, JointBertEncoder
from onir import vocab, util, config

from utils.aggregations import get_aggregation_func


@vocab.register('bert_qqa')
class BertVocabQQA(BertVocab):

    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.enc_aggregation = config['aggregation']

    @staticmethod
    def default_config():
        result = BertVocab.default_config().copy()
        result.update({
            'aggregation': 'mean'
        })
        return result

    def encoder(self):
        return {
            'joint': JointBertEncoderQQA,
            'sep': SepBertEncoderQQA,
        }[self.config['encoding']](self)


class SepBertEncoderQQA(SepBertEncoder):

    def __init__(self, vocabulary):
        super().__init__(vocabulary)
        self.aggregation_func = get_aggregation_func(vocabulary.enc_aggregation)

    def _enc_spec(self) -> dict:
        result = super()._enc_spec()
        result.update({
            'joint_fields': [
                'cls', 'query', 'doc', 'cls_query', 'cls_doc',
                'question', 'answer', 'cls_question', 'cls_answer'
            ]
        })
        return result

    def enc_query_doc(self, **inputs):
        result = {}
        if 'query_tok' in inputs and 'question_tok' in inputs and 'answer_tok' in inputs:
            query_results, query_cls = self._forward(inputs['query_tok'], inputs['query_len'], seg_id=0)
            question_results, question_cls = self._forward(inputs['question_tok'], inputs['question_len'], seg_id=0)
            answer_results, answer_cls = self._forward(inputs['answer_tok'], inputs['answer_len'], seg_id=0)
            aggregated_cls = self._aggregate_cls_embeddings(query_cls, question_cls, answer_cls)
            result.update({
                'qqa_cls': aggregated_cls
            })

        if 'doc_tok' in inputs and 'doc_len' in inputs:
            doc_results, doc_cls = self._forward(inputs['doc_tok'], inputs['doc_len'], seg_id=1)
            result.update({
                'doc': doc_results,
                'doc_cls': doc_cls
            })
        return result

    def _aggregate_cls_embeddings(self, query_cls, question_cls, answer_cls):
        aggregation = self.aggregation_func(query_cls[-1], question_cls[-1], answer_cls[-1], dim=1)
        return aggregation


class JointBertEncoderQQA(JointBertEncoder):
    """
    Concat QQA and send it to BERT
    """

    def _concat_tokens(self, query_tok, question_tok, answer_tok):
        sep = torch.empty(len(query_tok), 1, device=query_tok.device, dtype=query_tok.dtype).fill_(self.SEP)
        return torch.cat((query_tok, sep, question_tok, sep, answer_tok), dim=1)

    def enc_query_doc(self, **inputs):
        query_tok, query_len = inputs['query_tok'], inputs['query_len']
        question_tok, question_len = inputs['question_tok'], inputs['question_len']
        answer_tok, answer_len = inputs['answer_tok'], inputs['answer_len']
        doc_tok, doc_len = inputs['doc_tok'], inputs['doc_len']

        query_tok_aggregated = self._concat_tokens(query_tok, question_tok, answer_tok)
        query_len_aggregated = query_len + question_len + answer_len + 2 # two additional sep tokens

        BATCH, QLEN = query_tok_aggregated.shape
        maxlen = self.bert.config.max_position_embeddings
        MAX_DOC_TOK_LEN = maxlen - QLEN - 3  # -3 [CLS] and 2x[SEP]

        doc_toks, sbcount = util.subbatch(doc_tok, MAX_DOC_TOK_LEN)
        doc_mask = util.lens2mask(doc_len, doc_tok.shape[1])
        doc_mask, _ = util.subbatch(doc_mask, MAX_DOC_TOK_LEN)

        query_toks = torch.cat([query_tok_aggregated] * sbcount, dim=0)
        query_mask = util.lens2mask(query_len_aggregated, query_toks.shape[1])
        query_mask = torch.cat([query_mask] * sbcount, dim=0)

        CLSS = torch.full_like(query_toks[:, :1], self.CLS)
        SEPS = torch.full_like(query_toks[:, :1], self.SEP)
        ONES = torch.ones_like(query_mask[:, :1])
        NILS = torch.zeros_like(query_mask[:, :1])
        toks = torch.cat([CLSS, query_toks, SEPS, doc_toks, SEPS], dim=1)
        mask = torch.cat([ONES, query_mask, ONES, doc_mask, ONES], dim=1)
        segment_ids = torch.cat([NILS] * (2 + QLEN) + [ONES] * (1 + doc_toks.shape[1]), dim=1)

        # Change -1 padding to 0-padding (will be masked)
        toks = torch.where(toks == -1, torch.zeros_like(toks), toks)

        result = self.bert(toks, segment_ids, mask)

        # extract relevant subsequences for query and doc
        query_results = [r[:BATCH, 1:QLEN + 1] for r in result]
        doc_results = [r[:, QLEN + 2:-1] for r in result]
        doc_results = [util.un_subbatch(r, doc_tok, MAX_DOC_TOK_LEN) for r in doc_results]

        cls_results = []
        for layer in range(len(result)):
            cls_output = result[layer][:, 0]
            cls_result = []
            for i in range(cls_output.shape[0] // BATCH):
                cls_result.append(cls_output[i * BATCH:(i + 1) * BATCH])
            cls_result = torch.stack(cls_result, dim=2).mean(dim=2)
            cls_results.append(cls_result)

        if self.vocab.config['last_layer']:
            query_results = query_results[-1]
            doc_results = doc_results[-1]
            cls_results = cls_results[-1]

        return {
            'query': query_results,
            'doc': doc_results,
            'cls': cls_results
        }
