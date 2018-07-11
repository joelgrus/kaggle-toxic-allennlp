from typing import List

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers import WordTokenizer
from allennlp.models import Model
from allennlp.predictors import Predictor


@Predictor.register('toxic')
class ToxicPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = WordTokenizer()

    def _json_to_instance(self, json: JsonDict) -> Instance:
        # We're overriding `predict_json` directly, so we don't need this.  But I'd rather have a
        # useless stub here then make the base class throw a RuntimeError instead of a
        # NotImplementedError - the checking on the base class is worth it.
        raise RuntimeError("this should never be called")

    def predict_json(self, inputs: JsonDict, cuda_device: int = -1) -> JsonDict:
        batch_json = [inputs]
        batch_predictions = self.predict_batch_json(batch_json, cuda_device)
        return batch_predictions[0]

    def predict_batch_json(self, inputs: List[JsonDict], cuda_device: int = -1) -> List[JsonDict]:
        instances = [
            self._dataset_reader.text_to_instance(input['text'])
            for input in inputs
        ]

        outputs = self._model.forward_on_instances(instances, cuda_device)

        for input, output in zip(inputs, outputs):
            output['id'] = input['id']

        return sanitize(outputs)
