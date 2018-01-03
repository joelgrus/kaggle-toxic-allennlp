"""
Reads the toxic data dataset from a csv, where the data looks like

comment_id,comment_text,toxic,severe_toxic,obscene,threat,insult,identity_hate

where the last 6 label columns are all 0 or 1
(and where a comment can have multiple labels)
"""
from typing import List, Dict
import csv
import sys

import tqdm
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import DatasetReader, Dataset, Instance
from allennlp.data.fields import TextField, LabelField, ListField
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

# One of the fields is really long
csv.field_size_limit(sys.maxsize)

@DatasetReader.register('toxic')
class ToxicReader(DatasetReader):
    """
    toxic
    """
    def __init__(self,
                 max_length: int = None,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        self.max_length = max_length
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @classmethod
    def from_params(cls, params: Params) -> 'ToxicReader':
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        max_length = params.pop('max_length', None)
        params.assert_empty(cls.__name__)
        return cls(max_length=max_length, tokenizer=tokenizer, token_indexers=token_indexers)

    def read(self, file_path: str) -> Dataset:
        instances = []
        with open(file_path, "r") as data_file:
            reader = csv.reader(data_file)
            for row in tqdm.tqdm(reader):
                _, text, *labels = row
                instances.append(self.text_to_instance(text, labels))
        if not instances:
            raise ConfigurationError("No instances read!")

        return Dataset(instances)

    # pylint: disable=arguments-differ
    def text_to_instance(self,
                         text: str,
                         labels: List[str] = None) -> Instance:
        # There is a pathological example in the test set.
        if self.max_length is not None:
            text = text[:self.max_length]
        tokenized_text = self._tokenizer.tokenize(text)
        text_field = TextField(tokenized_text, self._token_indexers)

        fields = {'text': text_field}

        # Normally we wouldn't do this, but we need the test instances to have
        # the same "shape" as the train instances so that we can combine them
        # all into a single dataset.
        if not labels:
            labels = [0, 0, 0, 0, 0, 0]

        toxic, severe_toxic, obscene, threat, insult, identity_hate = labels

        # Because the labels are already 0 or 1, skip_indexing.
        fields['labels'] = ListField([
            LabelField(int(toxic),         skip_indexing=True),
            LabelField(int(severe_toxic),  skip_indexing=True),
            LabelField(int(obscene),       skip_indexing=True),
            LabelField(int(threat),        skip_indexing=True),
            LabelField(int(insult),        skip_indexing=True),
            LabelField(int(identity_hate), skip_indexing=True)
        ])

        return Instance(fields)
