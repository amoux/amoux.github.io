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

The code snippet below is a small demo of some of the classes currently built in `david`.

> NOTE: *The project is currently in private, feel free to contact me to know more!*

```python
import numpy as np
from david.models import GloVe
from david.tokenizers import Tokenizer
from david.datasets import YTCommentsDataset

# load preprocessed and clean dataset on multiple video categories
dataset, _ = YTCommentsDataset.split_train_test(4000)

# construct a Tokenizer object
tokenizer = Tokenizer(document=dataset,
                      remove_urls=True,
                      reduce_length=True)
                      
# aling the vocabulary in relation to the term count/frequency
tokenizer.fit_vocabulary(mincount=1)
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
from david.datasets import YTCommentsDataset

from david_sentiment import YTCSentimentConfig, YTCSentimentModel
import david_sentiment.dataset as ds

config = YTCSentimentConfig(project_dir="ytc_sentiment",
                            max_strlen=3000,
                            epochs=100,
                            enforce_ascii=True,
                            remove_urls=True,
                            glove_ndim="100d",)  

train_data, test_data = YTCommentsDataset.split_train_test(3000, subset=0.8)
x_train, y_labels, y_test = ds.fit_batch_to_dataset(train_data, config=config)

# train the embedding model
sm = YTCSentimentModel(config)
sm.train_model(x_train, y_labels)

# save everything - this eliminates the necessity to
# preprocess, tokenize and train everything again
sm.save_project()
```

**sentiment-outputs** : Below is an example of how `TextBlob` sentiment method fails to recognize any polarity from the following texts (from youtube comments). 

```
💬 (Textblob=0.0, YTCSentimentModel=61.0681)
  😑 - A BIG DEAL

💬 (Textblob=0.0, YTCSentimentModel=78.5567)
  😁 - Will you make a video on it ?

💬 (Textblob=0.0, YTCSentimentModel=94.1769)
  🤗 - we could hope to see in 2020??

💬 (Textblob=0.0, YTCSentimentModel=47.5426)
  😶 - Think about that.

💬 (Textblob=0.0, YTCSentimentModel=97.6973)
  😍 - Health, wealth and mind.
```

- Load and continue where you left off (both model & tokenizer instances are fully loaded):

```python
from david_sentiment import YTCSentimentConfig, YTCSentimentModel

config = YTCSentimentConfig.load_project('ytc_sentiment/config.ini')
ytc_sentiment = YTCSentimentModel(config)

print(ytc_sentiment)
'<YTCSentimentModel(max_seqlen=62, vocab_shape=(2552, 100))>'
...
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
