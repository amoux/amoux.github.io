## Portfolio

- ***Casual Interests***:
  - I use Linux at home and work.
  - I like building and fixing any type of computer as a hobby. (I know most devs dont but you can ask me to fix your laptop, computer or server).
  
  **Serious Interests***
  - Leveraging information extraction technology, particularly in NLP, to help people get answers to questions on broad and unstructured domains.
  - Building data pipelines (IN => OUT).
  - Implementing ML models to production pipelines.

Most recent code achievements `david.tokenizer.Tokenizer()` 😍 - used in all three projects below!

---

### I'm currently working on the following projects:

## david 

**nlp-toolkit**

> NLP toolkit for speeding data preparation on raw (dirty-data 🤮) social-media text.

- features:
  - Text-pipelines
  - Text preprocessing, cleaning, normalization & extraction utilities
  - Social-media text-scrapers (currently only youtube-comments)
  - Encoding/decoding tokenization based architecture (really happy about this one!)
  - Loading/downloading pre-trained models and datasets utilities, e.g., GloVe, BERT, ..ect.

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

## david-sentiment

**Easily train unsupervised sentiment models of any size on any text with a few lines of code**

> Below is a snippet of training an unsupervised sentiment model from scratch on comments scraped on YouTube.

```python
from david_sentiment import YTCSentimentConfig, YTCSentimentModel
import david_sentiment.dataset as ds

config = YTCSentimentConfig(project_dir="my-model", # project name
                            max_strlen=3000,
                            epochs=100,
                            enforce_ascii=True,
                            remove_urls=True,
                            glove_ndim="100d")

train_data, test_data = ds.ytcomments.split_train_test(3000, subset=0.8)
x_train, y_labels, y_test = ds.fit_batch_to_dataset(train_data, config=config)

# train the embedding model
sm = YTCSentimentModel(config)
sm.train_model(x_train, y_labels)

# save everything - this eliminates the necessity to
# preprocess, tokenize and train everything again
sm.save_project()
```

- **Notes on sentiment scores** : In the following output is an example of how `TextBlob's` ***rule-based*** sentiment method fails to recognize any polarity. On the other hand, the trained `CNN-LSTM` sentiment model gives us a spectrum of sentiments scores 👻

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

config = YTCSentimentConfig.load_project('my-model/config.ini')
sm = YTCSentimentModel(config)
print(sm)
...
'<YTCSentimentModel(max_seqlen=62, vocab_shape=(2552, 100))>'
```
---

## QAAM

**Question Answering Auto Model**

> Automatic question answering engine from any text source.

How about asking questions to any blog or website with texts instead of manually having to search and scroll to find exactly what you are looking for?

- unique-features:
  - Both document-similarity and query-correction algorithms used are language-independent - meaning that they work for any language.

- todos:
  - Adding support for all languages compatible with both `spaCy` and `Transformers`.
  - Cython implementation (currently testing 🤗)

```python
from qaam import QAAM
qaam = QAAM(0.2, metric='cosine', mode='tfidf')

blog_url = ("https://medium.com/analytics-vidhya/semantic-"
            "similarity-in-sentences-and-bert-e8d34f5a4677")
 
# after passing the URL - the text content is properly
# extracted, cleaned, and tokenized for you automatically!
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
```
<img src="images/pred2.png?raw=true"/>

- A word like food is correct, but it is not correct in terms of the document's context. Therefore, the word is automatically corrected to the most likely word based on the document's vocabulary.

```python
question = "Why is it food to use pre-trained sentencr encoters?"
qaam.answer(question, render=True)
```

<img src="images/pred1.png?raw=true"/>

---

### FAQ

- Can you help me with the frontend? No.
- Can you help me with the backend? Yes.
- Can you use library `x`? If its on python then yes `100%`
- What about using other languages? If it gets the job done the I will use it.

---

---
<p style="font-size:11px"></p>
<!-- Remove above link if you don't want to attibute -->
