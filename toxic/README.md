SUBMISSION
----------
https://medium.com/@nokkk/make-your-kaggle-submissions-with-kaggle-official-api-f49093c04f8a

kaggle competitions submissions -c jigsaw-toxic-comment-classification-challenge

https://www.google.com.au/search?q=kaggle+web+text+classification


TOOLS
=====
source activate py36
source deactivate


Architecture
------------
All classifier classes  must provide two methods.

    def fit(self, train):

    def predict(self, test):


### Examples
ClfGloveNBSpacy clf_glove_nb_spacy.py

TODO
====
Higher dropout. Validation loss is minimizing in 1st or 2nd epoch
    How to do dropout for LSTM? Smerity?
    Input vs recurrent dropout https://machinelearningmastery.com/use-dropout-lstm-networks-time-series-fore
    casting/
    other regularisations
    batch normalization
Use text length as a feature
SpaCY embeddings
Embedding width > 50
Twitter embeddings
Change probability clipping
Early stopping based on ROC
Select classifier based on column
Try xgboost
character embeddings
data augmentation https://arxiv.org/pdf/1502.01710.pdf
stratify keras validation split
use 2 validation sets and average XX
increase validation set size for better stopping
train:test => 80:20
try turning off trainable embeddings
multitask learning
   https://www.google.com.au/search?q=Stanford+Twitter+Sentiment+Corpus&oq=Stanford+Twitter+Sentiment+Corpus&aqs=chrome..69i57j0.1316j0j7&sourceid=chrome&ie=UTF-8
transfer learning (jhoward)

Need to clip
             |        toxic | severe_toxic |      obscene |       threat |       insult | identity_hate
------------ + ------------ + ------------ + ------------ + ------------ + ------------ + ------------
         min |       0.0000 |       0.0000 |       0.0000 |       0.0000 |       0.0000 |       0.0000
        mean |       0.0901 |       0.0129 |       0.0507 |       0.0020 |       0.0376 |       0.0096
         max |       1.0000 |       0.8277 |       0.9997 |       0.9047 |       0.9859 |       0.9465
------------ + ------------ + ------------ + ------------ + ------------ + ------------ + ------------

Clipping from 1.0 down has little effect
31915/31915 [==============================] - 116s 4ms/step
    0: 0: delta=     0 auc=0.97141 (toxic:0.966, severe_toxic:0.987, obscene:0.982, threat:0.951, insult:0.978, identity_hate:0.964)
    0: 1: delta= 1e-06 auc=0.97141 (toxic:0.966, severe_toxic:0.987, obscene:0.982, threat:0.951, insult:0.978, identity_hate:0.964)
    0: 2: delta= 1e-05 auc=0.97141 (toxic:0.966, severe_toxic:0.987, obscene:0.982, threat:0.951, insult:0.978, identity_hate:0.964)
    0: 3: delta=0.0001 auc=0.97141 (toxic:0.966, severe_toxic:0.987, obscene:0.982, threat:0.951, insult:0.978, identity_hate:0.964)
    0: 4: delta= 0.001 auc=0.97141 (toxic:0.966, severe_toxic:0.987, obscene:0.982, threat:0.951, insult:0.978, identity_hate:0.964)
    0: 5: delta=  0.01 auc=0.97141 (toxic:0.966, severe_toxic:0.987, obscene:0.982, threat:0.951, insult:0.978, identity_hate:0.964)
    0: 6: delta=   0.1 auc=0.97138 (toxic:0.966, severe_toxic:0.987, obscene:0.982, threat:0.951, insult:0.978, identity_hate:0.964)
    0: 7: delta=   0.2 auc=0.97130 (toxic:0.966, severe_toxic:0.987, obscene:0.982, threat:0.951, insult:0.978, identity_hate:0.964)
    0: 8: delta=   0.3 auc=0.97117 (toxic:0.965, severe_toxic:0.987, obscene:0.982, threat:0.951, insult:0.978, identity_hate:0.964)
    0: 9: delta=   0.5 auc=0.97072 (toxic:0.964, severe_toxic:0.987, obscene:0.981, threat:0.951, insult:0.977, identity_hate:0.964)
    0: 10: delta=   0.8 auc=0.96867 (toxic:0.960, severe_toxic:0.986, obscene:0.978, threat:0.951, insult:0.973, identity_hate:0.964)
    0: 11: delta=   0.9 auc=0.96593 (toxic:0.954, severe_toxic:0.984, obscene:0.975, threat:0.951, insult:0.968, identity_hate:0.963)
    0: auc=0.971 (toxic:0.966, severe_toxic:0.987, obscene:0.982, threat:0.951, insult:0.978, identity_hate:0.964)
stats=[3, 6]
