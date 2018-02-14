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
Higher dropout. Validation loss is minimizing in 1st or 2nd epoch  How to do dropout for LSTM? Smerity?
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
use 2 validation sets and average
