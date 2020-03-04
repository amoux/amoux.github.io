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

```python
from david.tokenizer import Tokenizer
from david.youtube import YTCommentScraper

scraper = YTCommentScraper()
iterbatch = scraper.scrape_comments_generator(video_id='<VIDEO_ID>')
document = [comment['text'] for comment in iterbatch]

tokenizer = Tokenizer(document=document)
tokenizer.fit_vocabulary(mincount=2)
sequences = tokenizer.document_to_sequences(document)
tfidf_matrix = tokenizer.sequences_to_matrix(sequences, 'tfidf')

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
qaam.texts_from_url('<WEBSITE_URL>')
qaam.texts_from_doc(iterable_document)
qaam.texts_from_str(string_sequences)

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
