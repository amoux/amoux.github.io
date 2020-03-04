## Portfolio

---

### I'm currently working on:

***david nlp-toolkit*** : ( `in-progress` )

> NLP toolkit for speeding data preparation for social-media texts.

- features:
  - text-pipelines
  - text preprocessing, cleaning, normalization & extraction utilities
  - social-media text-scrapers (currently only youtube-comments)
  - encoding-decoding tokenization based architecture (really happy about this one!)
  - loading/downloading pre-trained models and datasets utilities, e.g., GloVe, BERT, ..ect.

The code snippet below is a small demo of some of the classes currently built (The project is currently in private, contact me to know more!)

```python
import numpy as np
from david.models import GloVe
from david.tokenizers import Tokenizer
from david.datasets import YTCommentsDataset

# load preprocessed and clean dataset on multiple video categories
dataset, _ = YTCommentsDataset.split_train_test(4000)

# construct a Tokenizer object
tokenizer = Tokenizer(remove_urls=True, reduce_length=True)
tokenizer.fit_on_document(document=dataset)
tokenizer.vocabulary_to_frequency(mincount=2)

# embedd the vocabulary
vocab_matrix = GloVe.fit_embeddings(tokenizer.vocab_index, vocab_dim="100d")

def most_similar(word: str, k=5):
    """Fetch a word-query, retrieving the top most similar tokens."""
    embedding = vocab_matrix[tokenizer.convert_string_to_ids(word)[0]]
    dst = (np.dot(vocab_matrix, embedding)
           / np.linalg.norm(vocab_matrix, ord=None, axis=1)
           / np.linalg.norm(embedding)))
    token_ids = np.argsort(-dst)
    id2tok = {idx: tok for tok, idx in tokenizer.vocab_index.items()}
    return [(id2tok[i], dst[i]) for i in token_ids if i in id2tok][1: k+1]
    
    
most_similar("google", k=7)  # most similar tokens to google.
[('facebook', 0.7516581668453545),
 ('internet', 0.7383222858698717),
 ('online', 0.6866507281595066),
 ('users', 0.6830479303146789),
 ('software', 0.6750386018261412),
 ('twitter', 0.6647332902232169),
 ('youtube', 0.6424136902092844)]

most_similar("comment", k=7)  # most similar tokens to comment
[('statement', 0.7802159956616586),
 ('comments', 0.7776707404754829),
 ('details', 0.7315377076004379),
 ('asked', 0.7290169506697899),
 ('saying', 0.7005583039383192),
 ('statements', 0.6972063046229513),
 ('suggestion', 0.6785569811078331)]
```
---

***david-sentiment*** : ( `in-progress` )

> Training unsupervised sentiment models from social-media texts.

```python
from david_sentiment import YTCSentimentConfig, YTCSentimentModel
config = YTCSentimentConfig.load_project('my-model/config.ini')
sm = YTCSentimentModel(config)
sm.train_model()
sm.save_project()
```
---

***QAAM***  : ( `in-progress` )

> Automatic question answering engine from any text source.

```python
from qaam import QAAM
qaam = QAAM(0.2, metric='cosine', mode='tfidf')

# after loading the texts - an instace of the all
# the texts in the document or website is created
blog_url = ("https://medium.com/analytics-vidhya/semantic-"
            "similarity-in-sentences-and-bert-e8d34f5a4677")
qaam.texts_from_url(blog_url)

# obtain all the entities from the document
entities = qaam.common_entities(None, lower=True, lemma=True)
...
[('bert', 14),
 ('nlp', 8),
 ('google', 3),
 ('glove', 2),
 ('universal sentence encoder', 2), ...]
 ```
 
- Here is an example of how `BERTO` is corrected to the proper context-term `BERT` - before computing document similarity and passing the context to the `Tranformers Auto Model` for question answering.

```python
qaam.answer("How was BERTO trained?", render=True)
...
```
<img src="images/pred2.png?raw=true"/>

---

### Other projects

- [Project 1 Title](http://example.com/)
- [Project 2 Title](http://example.com/)
- [Project 3 Title](http://example.com/)
- [Project 4 Title](http://example.com/)
- [Project 5 Title](http://example.com/)

---

---
<p style="font-size:11px"></p>
<!-- Remove above link if you don't want to attibute -->
