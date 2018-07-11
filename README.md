# Toxic Comment Classification Challenge

Over the winter break I decided to take a crack at using [AllenNLP](http://allennlp.org)
for the [Kaggle Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).
My goal was not to win, but just to see how hard it would
be to use AllenNLP for a "novel" problem.
("Novel" only in the sense that I don't think we've tried AllenNLP for a Kaggle contest before.)

# The Challenge

The challenge is a [multi-label classification](https://en.wikipedia.org/wiki/Multi-label_classification) problem.
Each training example consists of a comment id, a wikipedia comment, and a binary vector of length 6
indicating whether the example contains a certain type of toxicity.

# Data

The training data is in CSV format, so I created a custom `ToxicReader` subclass of `DatasetReader`
that indexes the comment in a `TextField` and then creates a `ListField` of `LabelField`s for the 6 labels.

It consists of about 96k examples. I split these into roughly 85k training examples and 11k validation examples.

The test data is also a CSV but only has the comment_id and comment columns.

As this is a Kaggle competition and not an academic reference dataset, a good entry would spend a lot of time
cleaning and massaging the data. I didn't do this.

# The Model

I largely followed MattG's [Using AllenNLP in your Project](https://github.com/allenai/allennlp/blob/v0.3.0/tutorials/getting_started/using_in_your_repo.md)
tutorial, and my model is a slight variant on his:

* tokenize the comment
* embed the tokens using 100d GloVe vectors (I later found a Kaggle discussion about how this embedding performs poorly on this task)
* use a bidirectional LSTM to encode each comment as a 200d vector
* produce 6 output logits using a 2-layer feedforward network
* apply `torch.nn.functional.sigmoid` to the output logits to get 6 probabilities

I trained the model using `torch.nn.MultiLabelSoftMarginLoss` (on the logits) and implemented a `MultilabelF1`
to use as the validation metric.

# The Result

This model doesn't do great, right at this moment I am in 750th place (out of about 1000).
My guess is that to improve things I'd need to invest time in three areas:

(1) cleaning the data
(2) doing some feature engineering
(3) coming up with better word embeddings

None of these is on my short-term to-do list, so probably I'm stuck near the bottom of the leaderboard.

# A Few Issues

Part of why I did this was to see what issues came up when I took AllenNLP out of its comfort zone.
I found a couple:

(1) here the test data is unlabeled, which means it has a different "shape" from the training data,
 which means that (as the library works currently) it's not possible to combine test and training data
 into a single dataset (which I needed to do to get the test-vocab word embeddings into the model).
 We have some long-term plans to fix this (and broader issues). As a short-term workaround I just made up
 labels for the test data, which didn't affect anything.

(2) our `Predictor` workflow is premised on JSON -> prediction -> JSON. Here our test data is in CSV format,
 which makes it hard to generate the submission containing the test data and predictions (which also needs to be in CSV format).
 I made a PR to fix this, but in the meantime I just wrote my own `predict.py` that deals natively with CSVs.

# Running it Yourself

I just (July 2018) updated it to work with the latest released allennlp (0.5.1).

To run the code, create a new virtual environment and

```
pip install allennlp==0.5.1
```

Then you can train the model using the command

```
allennlp train --include-package toxic baseline.json -s /tmp/your/serialization/directory
```

(The `include-package` flag is necessary for the allennlp code to know about the
 custom classes defined here.)

You might need to do `export PYTHONPATH=.`, I have been doing Python for years
and still don't really understand when that's necessary.
