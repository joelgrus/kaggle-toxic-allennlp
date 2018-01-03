"""
Because ``allennlp.run predict`` is designed to do
JSON-in-JSON-out, it doesn't suit our CSV-in-CSV-out
use case. I have an idea how to fix it, but in the meantime
I just copied the logic into my own ``predict`` script.
"""
# pylint: disable=invalid-name,unused-import

import csv
import sys

from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor

from toxic.data.dataset_reader.reader import ToxicReader
from toxic.models.model import ToxicModel
from toxic.service.predictors.predictor import ToxicPredictor

ARCHIVE_FILE = 'serialization/baseline/model.tar.gz'
CUDA_DEVICE = -1
PREDICTOR_NAME = 'toxic'
INPUT_FILE = 'data/test'
OUTPUT_FILE = 'serialization/baseline/predictions.csv'
PRINT_TO_CONSOLE = False
BATCH_SIZE = 128

if __name__ == "__main__":
    archive = load_archive(ARCHIVE_FILE, CUDA_DEVICE)
    predictor = Predictor.from_archive(archive, PREDICTOR_NAME)
    with open(INPUT_FILE, 'r') as input_file, \
         open(OUTPUT_FILE, 'w') as output_file:

        writer = csv.writer(output_file)
        writer.writerow(["id","toxic","severe_toxic","obscene","threat","insult","identity_hate"])

        def _run_predictor(batch_data):
            if len(batch_data) == 1:
                result = predictor.predict_json(batch_data[0], CUDA_DEVICE)
                # Batch results return a list of json objects, so in
                # order to iterate over the result below we wrap this in a list.
                results = [result]
            else:
                results = predictor.predict_batch_json(batch_data, CUDA_DEVICE)

            for model_input, output in zip(batch_data, results):

                row = [output['id']] + output['probabilities']

                if PRINT_TO_CONSOLE:
                    print("input: ", model_input)
                    print("prediction: ", row)
                writer.writerow(row)

        batch_json_data = []
        for comment_id, text in csv.reader(input_file):
            json_data = {'id': comment_id, 'text': text}
            batch_json_data.append(json_data)
            if len(batch_json_data) == BATCH_SIZE:
                _run_predictor(batch_json_data)
                batch_json_data = []

        # We might not have a dataset perfectly divisible by the batch size,
        # so tidy up the scraps.
        if batch_json_data:
            _run_predictor(batch_json_data)
