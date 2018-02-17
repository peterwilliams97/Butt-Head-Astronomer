SUBMISSION
----------
https://medium.com/@nokkk/make-your-kaggle-submissions-with-kaggle-official-api-f49093c04f8a

kaggle competitions submissions -c jigsaw-toxic-comment-classification-challenge

https://www.google.com.au/search?q=kaggle+web+text+classification


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
transfer learning (jhoward)

Need to clip
             |        toxic | severe_toxic |      obscene |       threat |       insult | identity_hate
------------ + ------------ + ------------ + ------------ + ------------ + ------------ + ------------
         min |       0.0000 |       0.0000 |       0.0000 |       0.0000 |       0.0000 |       0.0000
        mean |       0.0901 |       0.0129 |       0.0507 |       0.0020 |       0.0376 |       0.0096
         max |       1.0000 |       0.8277 |       0.9997 |       0.9047 |       0.9859 |       0.9465
------------ + ------------ + ------------ + ------------ + ------------ + ------------ + ------------
