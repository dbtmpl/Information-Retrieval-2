from onir.vocab import BertVocab
from onir.vocab.bert_vocab import SepBertEncoder, JointBertEncoder
from onir import vocab, util, config


@vocab.register('bert_qqa')
class BertVocabQQA(BertVocab):

    def encoder(self):
        return {
            'joint': JointBertEncoderQQA,
            'sep': SepBertEncoderQQA,
        }[self.config['encoding']](self)


class SepBertEncoderQQA(SepBertEncoder):

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
        if 'query_tok' in inputs and 'query_len' in inputs:
            query_results, query_cls = self._forward(inputs['query_tok'], inputs['query_len'], seg_id=0)
            result.update({
                'query': query_results,
                'query_cls': query_cls
            })
        if 'question_tok' in inputs and 'question_len' in inputs:
            question_results, question_cls = self._forward(inputs['question_tok'], inputs['question_len'], seg_id=0)
            result.update({
                'question': question_results,
                'question_cls': question_cls
            })
        if 'answer_tok' in inputs and 'answer_len' in inputs:
            answer_results, answer_cls = self._forward(inputs['answer_tok'], inputs['answer_len'], seg_id=0)
            result.update({
                'answer': answer_results,
                'answer_cls': answer_cls
            })

        if 'doc_tok' in inputs and 'doc_len' in inputs:
            doc_results, doc_cls = self._forward(inputs['doc_tok'], inputs['doc_len'], seg_id=1)
            result.update({
                'doc': doc_results,
                'cls': doc_cls
            })
        return result


class JointBertEncoderQQA(JointBertEncoder):
    pass
