from onir import trainers
from onir.trainers import PairwiseTrainer

from utils.general_utils import apply_spec_batch_qqa


@trainers.register('pairwise_qqa')
class PairwiseTrainerQQA(PairwiseTrainer):

    def iter_batches(self, it):
        while True:  # breaks on StopIteration
            input_data = {}
            for _, record in zip(range(self.batch_size), it):
                for k, v in record.items():
                    assert len(v) == self.numneg + 1
                    for seq in v:
                        input_data.setdefault(k, []).append(seq)
            input_data = apply_spec_batch_qqa(input_data, self.input_spec, self.device)
            yield input_data
