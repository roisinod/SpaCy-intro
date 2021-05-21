import pandas as pd
import spacy
from spacy.lang.en import examples
from spacy.util import minibatch
from spacy.pipeline.textcat import single_label_cnn_config
import random
from spacy.training.example import Example
from spacy.pipeline.textcat_multilabel import DEFAULT_MULTI_TEXTCAT_MODEL
df = pd.read_csv('spam dataset.csv .csv')
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
#print(df.columns)
#print(df.head(10))
#create blank model
nlp = spacy.blank("en")
config = {
   "threshold": 0.5,
   "model": DEFAULT_MULTI_TEXTCAT_MODEL,
}

textcat = nlp.add_pipe("textcat", config=config)
textcat.add_label("ham")
textcat.add_label("spam")
train_texts = df['text'].values
train_labels = [{'cats': {'ham': label == 'ham',
                          'spam': label == 'spam'}}
                for label in df['label']]
train_data = list(zip(train_texts, train_labels))
print(train_data[:3])
spacy.util.fix_random_seed(1)
optimizer = nlp.initialize()
# Create the batch generator with batch size = 8
batches = minibatch(train_data, size=8)

# Iterate through minibatches
TRAIN_DATA = train_data
random.shuffle(TRAIN_DATA)
losses = {}
for batch in minibatch(TRAIN_DATA, size=8):
    for text, annotations in batch:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], drop=0.35, sgd=optimizer, losses=losses)

import random

random.seed(1)
spacy.util.fix_random_seed(1)
optimizer = nlp.begin_training()

losses = {}
for epoch in range(10):
    random.shuffle(train_data)
    # Create the batch generator with batch size = 8
    batches = minibatch(train_data, size=8)
    # Iterate through minibatches
    for batch in batches:
        for text, annotations in batch:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], drop=0.35, sgd=optimizer, losses=losses)
    #print(losses)
texts = ["Are you ready for the tea party????? It's gonna be wild",
         "URGENT Reply to this message for GUARANTEED FREE TEA"]
docs = [nlp.tokenizer(text) for text in texts]

# Use textcat to get the scores for each doc
textcat = nlp.get_pipe('textcat')
scores, _ = textcat.predict(docs)

print(scores)
predicted_labels = scores.argmax(axis=1)
print([textcat.labels[label] for label in predicted_labels])