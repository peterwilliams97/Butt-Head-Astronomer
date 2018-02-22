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
http://forums.fast.ai/t/language-model-in-thai/9874
SpaCy
https://arxiv.org/abs/1802.00385
AllenNlp Elmo
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

instance 2
   0: auc=0.982 [0.9750561  0.98830847 0.98900364 0.97884294 0.98374052 0.9796811 ] (200, 0.1, 30000, (0.001, 0.001, 0.002, 0.003, 0.0), 150, 1, '6B', 300) ClfLstmGlove(batch_size=64, dropout=0.1, embed_name=6B, embed_size=300, epochs=40, learning_rate=(0.001, 0.001, 0.002, 0.003, 0.0), max_features=30000, maxlen=150, n_folds=1, n_hidden=200)
   1: auc=0.981 [0.97351159 0.98882149 0.98829532 0.97717715 0.98304252 0.97756941] (200, 0.1, 30000, (0.001, 0.001, 0.002, 0.003, 0.0), 150, 1, '840B', 300) ClfLstmGlove(batch_size=64, dropout=0.1, embed_name=840B, embed_size=300, epochs=40, learning_rate=(0.001, 0.001, 0.002, 0.003, 0.0), max_features=30000, maxlen=150, n_folds=1, n_hidden=200)
   2: auc=0.981 [0.96983904 0.98820016 0.98936429 0.97642246 0.98377134 0.97741031] (200, 0.1, 30000, (0.001, 0.001, 0.002, 0.003, 0.0), 150, 2, '840B', 300) ClfLstmGlove(batch_size=64, dropout=0.1, embed_name=840B, embed_size=300, epochs=40, learning_rate=(0.001, 0.001, 0.002, 0.003, 0.0), max_features=30000, maxlen=150, n_folds=1, n_hidden=200)
   3: auc=0.980 [0.97371822 0.98834821 0.98607619 0.97394626 0.98195597 0.97681372] (200, 0.1, 20000, (0.001, 0.001, 0.002, 0.003, 0.0), 150, 1, '840B', 300) ClfLstmGlove(batch_size=64, dropout=0.1, embed_name=840B, embed_size=300, epochs=40, learning_rate=(0.001, 0.001, 0.002, 0.003, 0.0), max_features=20000, maxlen=150, n_folds=1, n_hidden=200)
   4: auc=0.980 [0.9757477  0.98809623 0.98858001 0.97236212 0.98183801 0.97235197] (200, 0.1, 30000, (0.001, 0.001, 0.002, 0.003, 0.0), 200, 1, '840B', 300) ClfLstmGlove(batch_size=64, dropout=0.1, embed_name=840B, embed_size=300, epochs=40, learning_rate=(0.001, 0.001, 0.002, 0.003, 0.0), max_features=30000, maxlen=200, n_folds=1, n_hidden=200)
   5: auc=0.979 [0.97508761 0.98684306 0.9888123  0.96740227 0.98302377 0.97569144] (200, 0.2, 30000, (0.001, 0.001, 0.002, 0.003, 0.0), 150, 1, '840B', 300) ClfLstmGlove(batch_size=64, dropout=0.2, embed_name=840B, embed_size=300, epochs=40, learning_rate=(0.001, 0.001, 0.002, 0.003, 0.0), max_features=30000, maxlen=150, n_folds=1, n_hidden=200)
   6: auc=0.979 [0.97140636 0.98771753 0.98595271 0.97324315 0.98139698 0.97699946] (200, 0.1, 30000, (0.001, 0.001, 0.002, 0.003, 0.0), 150, 1, '6B', 200) ClfLstmGlove(batch_size=64, dropout=0.1, embed_name=6B, embed_size=200, epochs=40, learning_rate=(0.001, 0.001, 0.002, 0.003, 0.0), max_features=30000, maxlen=150, n_folds=1, n_hidden=200)
   7: auc=0.979 [0.97363074 0.98655193 0.98809387 0.97010162 0.98237335 0.97454368] (200, 0.1, 30000, (0.002, 0.002, 0.002, 0.003, 0.0), 150, 1, '840B', 300) ClfLstmGlove(batch_size=64, dropout=0.1, embed_name=840B, embed_size=300, epochs=40, learning_rate=(0.002, 0.002, 0.002, 0.003, 0.0), max_features=30000, maxlen=150, n_folds=1, n_hidden=200)
   8: auc=0.978 [0.96968342 0.98657204 0.98447823 0.97598084 0.97960141 0.97371561] (200, 0.1, 30000, (0.001, 0.001, 0.002, 0.003, 0.0), 70, 1, '840B', 300) ClfLstmGlove(batch_size=64, dropout=0.1, embed_name=840B, embed_size=300, epochs=40, learning_rate=(0.001, 0.001, 0.002, 0.003, 0.0), max_features=30000, maxlen=70, n_folds=1, n_hidden=200)
   9: auc=0.978 [0.97274651 0.98564288 0.98787301 0.96730656 0.98123012 0.97244824] (200, 0.1, 30000, (0.001, 0.001, 0.002, 0.003, 0.0), 150, 3, '840B', 300) ClfLstmGlove(batch_size=64, dropout=0.1, embed_name=840B, embed_size=300, epochs=40, learning_rate=(0.001, 0.001, 0.002, 0.003, 0.0), max_features=30000, maxlen=150, n_folds=1, n_hidden=200)

instance 3
   0: auc=0.981 [0.97338312 0.9891582  0.98879899 0.97496824 0.98250596 0.97656324] (200, 0.1, 30000, (0.002, 0.002, 0.002, 0.003, 0.0), 150, 1, '840B', 300) ClfLstmGlove(batch_size=64, dropout=0.1, embed_name=840B, embed_size=300, epochs=40, learning_rate=(0.002, 0.002, 0.002, 0.003, 0.0), max_features=30000, maxlen=150, n_folds=1, n_hidden=200)
   1: auc=0.981 [0.97641332 0.98821867 0.98859288 0.97485211 0.98316937 0.97368016] (200, 0.1, 30000, (0.002, 0.002, 0.002, 0.003, 0.0), 150, 2, '840B', 300) ClfLstmGlove(batch_size=64, dropout=0.1, embed_name=840B, embed_size=300, epochs=40, learning_rate=(0.002, 0.002, 0.002, 0.003, 0.0), max_features=30000, maxlen=150, n_folds=1, n_hidden=200)
   2: auc=0.981 [0.9737291  0.98868542 0.98822773 0.97557122 0.98238556 0.97572784] (200, 0.1, 30000, (0.001, 0.001, 0.002, 0.003, 0.0), 150, 1, '840B', 300) ClfLstmGlove(batch_size=64, dropout=0.1, embed_name=840B, embed_size=300, epochs=40, learning_rate=(0.001, 0.001, 0.002, 0.003, 0.0), max_features=30000, maxlen=150, n_folds=1, n_hidden=200)
   3: auc=0.979 [0.9753885  0.98834315 0.98859767 0.96284708 0.98333834 0.97814435] (200, 0.2, 30000, (0.001, 0.001, 0.002, 0.003, 0.0), 150, 1, '840B', 300) ClfLstmGlove(batch_size=64, dropout=0.2, embed_name=840B, embed_size=300, epochs=40, learning_rate=(0.001, 0.001, 0.002, 0.003, 0.0), max_features=30000, maxlen=150, n_folds=1, n_hidden=200)
   4: auc=0.979 [0.97369484 0.98702978 0.9896832  0.96533211 0.98305046 0.97592929] (200, 0.1, 30000, (0.002, 0.002, 0.002, 0.003, 0.0), 150, 3, '840B', 300) ClfLstmGlove(batch_size=64, dropout=0.1, embed_name=840B, embed_size=300, epochs=40, learning_rate=(0.002, 0.002, 0.002, 0.003, 0.0), max_features=30000, maxlen=150, n_folds=1, n_hidden=200)
   5: auc=0.978 [0.97171631 0.98800955 0.98651588 0.96530701 0.9813796  0.97763309] (200, 0.1, 30000, (0.002, 0.002, 0.002, 0.003, 0.0), 200, 1, '840B', 300) ClfLstmGlove(batch_size=64, dropout=0.1, embed_name=840B, embed_size=300, epochs=40, learning_rate=(0.002, 0.002, 0.002, 0.003, 0.0), max_features=30000, maxlen=200, n_folds=1, n_hidden=200)
