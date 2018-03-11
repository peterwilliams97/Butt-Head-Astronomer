/home/pcadmin/code/Butt-Head-Astronomer/toxic/logs/spacy_lstm2.log

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
https://machinelearningmastery.com/machine-learning-performance-improvement-cheat-sheet/
https://github.com/facebookresearch/fastText#obtaining-word-vectors-for-out-of-vocabulary-words
http://forums.fast.ai/t/language-model-in-thai/9874
SpaCy
  Second classifier based on lemmas, POS, named entities
Add sentence remainders until maxlen is exhausted  <-- closer to bigrams
Reduce embedding dimensionaliyty. Which size is best for sentiment classification?
Smaller batch size

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
Try xgboost  e.g. For blending sentence vectors
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
try a 3 layer LSTM
try LSTM with smaller higher layers
search for reduction params (linear(lo1, hi1, by order in doc) + a * linear(lo2, hi2, by auc))

better tokenization
get_embeddings: oov=1163  "don't",  "i'm", 'it´s',  '20mins', '82.209.225.33', "i'll"  wikipedia!!hi', 354), ('bitches.fuck'

PROGRESS
========
gpu2: 40000 test
Mean: auc=0.975 (toxic:0.967, severe_toxic:0.984, obscene:0.982, threat:0.967, insult:0.974, identity_hate:0.973)
--------------------------------------------------------------------------------------------------------------
auc=0.975 +- 0.000 (0%) range=0.000 (0%)
program=trial_spacy9_submit_simple.py train=[39999, 8]
=========

gpu4: 20000 test n=3
 ClfSpacy(n_hidden=64, max_length=75, max_features=20000, # Shape
                    dropout=0.5, learn_rate=0.001, frozen=False, # General NN config
                    epochs=epochs, batch_size=300,
                    lstm_type=6, predict_method=PREDICT_METHODS_GOOD[0])
-------------------------------------------------------------------------------------------------------------- n=3
    0: auc=0.975 (toxic:0.964, severe_toxic:0.982, obscene:0.979, threat:0.989, insult:0.977, identity_hate:0.960)
    1: auc=0.965 (toxic:0.964, severe_toxic:0.989, obscene:0.982, threat:0.932, insult:0.977, identity_hate:0.944)
    2: auc=0.969 (toxic:0.960, severe_toxic:0.974, obscene:0.976, threat:0.961, insult:0.973, identity_hate:0.969)
 Mean: auc=0.970 (toxic:0.963, severe_toxic:0.982, obscene:0.979, threat:0.961, insult:0.976, identity_hate:0.958)
--------------------------------------------------------------------------------------------------------------
auc=0.970 +- 0.004 (0%) range=0.010 (1%)
program=trial_spacy9_submit_simple.py train=[19999, 8]
===========
gpu3: Parameter search 20000
gpu5: Parameter search 20000


Old PROGRESS
============
Give up lstm_type=10
Try: n_hidden=1000
Try: auc=0.9852   0: [0.974506 0.987315 0.988518 0.992702 0.9818   0.986232] ClfSpacy(batch=150, dropout=0.5, epochs=20, epochs2=2, frozen=True, lr=0.001, lstm_type=9, max_length=75, n_hidden=512, m=MEAN) epochs?

gpu2: trial_spacy11_submit.py submission ClfSpacy(n_hidden=512, max_length=100,
                    dropout=0.3, learn_rate=0.001,
                    epochs=6, batch_size=300, frozen=True,
                    lstm_type=6, predict_method=PREDICT_METHODS_GOOD[0])
gpu3: Testing many parameters on 40000
gpu4: Testing many parameters on 40000
gpu5: spacy_lstmx_90.ALL.submission  prog='trial_spacy9_submit.py' ClfSpacy(batch_size=300, dropout=0.5, epochs=9, epochs2=2, frozen=True, learn_rate=0.001, lstm_type=9, max_length=75, n_hidden=512, predict_method=MEAN) submission=True

gpu6: Testing many parameters on 40000

gpu7: spacy_lstmx_90.ALL.submission'  prog='trial_spacy9_submit.py' ClfSpacy(batch_size=300, dropout=0.5, epochs=9, epochs2=2, frozen=True, learn_rate=0.001, lstm_type=9, max_length=75, n_hidden=512, predict_method=MEAN) submission=False

instance-5: Testing many parameters on 40000


SUBMISSIONS
===========
gpu5: spacy_lstmx_90.ALL.LINEAR.csv
Your submission scored 0.9760, which is not an improvement of your best score. Keep trying!
spacy_lstmx_90.ALL.submission
ensembled with NB: ubmission scored 0.9805, which is not an improvement of your best score.

gpu2: spacy_lstmx_110.ALL.MEAN.csv
Your submission scored 0.9724, which is not an improvement of your best score. Keep trying!

RESULTS
=======

gpu3: spacy_lstm120_flip.40000.log
RESULTS SUMMARY: 10
auc=0.9852   9: get_clf42 ClfSpacy(batch_size=150, dropout=0.5, epochs=20, epochs2=2, frozen=True, learn_rate=0.001, lstm_type=9, max_length=75, n_hidden=512, predict_method=LINEAR)
auc=0.9842   8: get_clf42 ClfSpacy(batch_size=150, dropout=0.5, epochs=20, epochs2=2, frozen=True, learn_rate=0.001, lstm_type=9, max_length=75, n_hidden=512, predict_method=MEAN)
auc=0.9828   5: get_clf41 ClfSpacy(batch_size=300, dropout=0.5, epochs=20, epochs2=2, frozen=True, learn_rate=0.001, lstm_type=9, max_length=75, n_hidden=256, predict_method=LINEAR)
auc=0.9828   6: get_clf41 ClfSpacy(batch_size=300, dropout=0.5, epochs=20, epochs2=2,

instance-5: spacy_lstm22_flip.40000.log
RESULTS SUMMARY: 20
auc=0.9848   9: get_clf25 ClfSpacy(batch_size=300, dropout=0.5, epochs=20, epochs2=2, frozen=True, learn_rate=0.001, lstm_type=9, max_length=75, n_hidden=512, predict_method=LINEAR) best_epoch=6

auc=0.9848  10: get_clf25 ClfSpacy(batch_size=300, dropout=0.5, epochs=20, epochs2=2, frozen=True, learn_rate=0.001, lstm_type=9, max_length=75, n_hidden=512, predict_method=LINEAR2)
auc=0.9847  11: get_clf25 ClfSpacy(batch_size=300, dropout=0.5, epochs=20, epochs2=2, frozen=True, learn_rate=0.001, lstm_type=9, max_length=75, n_hidden=512, predict_method=LINEAR3)
auc=0.9844   8: get_clf25 ClfSpacy(batch_size=300, dropout=0.5, epochs=20, epochs2=2, frozen=True, learn_rate=0.001, lstm_type=9, max_length=75, n_hidden=512, predict_method=MEAN)
auc=0.9824   4: get_clf24 ClfSpacy(batch_size=100, dropout=0.1, epochs=20, epochs2=2, frozen=True, learn_rate=0.001, lstm_type=9, max_length=100, n_hidden=512, predict_method=MEAN)


gpu3: spacy_lstm21_flip.40000.log
RESULTS SUMMARY: 20
auc=0.9815   5: get_clf25 ClfSpacy(batch_size=300, dropout=0.5, epochs=20, epochs2=2, frozen=True, learn_rate=0.001, lstm_type=7, max_length=75, n_hidden=512, predict_method=LINEAR)
auc=0.9815   6: get_clf25 ClfSpacy(batch_size=300, dropout=0.5, epochs=20, epochs2=2, frozen=True, learn_rate=0.001, lstm_type=7, max_length=75, n_hidden=512, predict_method=LINEAR2)
auc=0.9814   7: get_clf25 ClfSpacy(batch_size=300, dropout=0.5, epochs=20, epochs2=2, frozen=True, learn_rate=0.001, lstm_type=7, max_length=75, n_hidden=512, predict_method=LINEAR3)

gpu3: spacy_lstm21_flip.40000.log
auc=0.9840   9: get_clf25 ClfSpacy(batch_size=300, dropout=0.5, learn_rate=0.001, lstm_type=9, max_length=75, n_hidden=512, pm=LINEAR)
auc=0.9840  10: get_clf25 ClfSpacy(batch_size=300, dropout=0.5, learn_rate=0.001, lstm_type=9, max_length=75, n_hidden=512, pm=LINEAR2)
auc=0.9839  11: get_clf25 ClfSpacy(batch_size=300, dropout=0.5, learn_rate=0.001, lstm_type=9, max_length=75, n_hidden=512, pm=LINEAR3)


RESULTS SUMMARY: 13 - 6:ALL MEAN 2
auc=0.9837   0: [0.972902 0.987188 0.988681 0.986164 0.981904 0.98516 ] ClfSpacy(batch=300, dropout=0.5, epochs=20, epochs2=2, frozen=True, lr=0.001, lstm_type=9, max_length=75, n_hidden=512, m=MEAN)
auc=0.9812   1: [0.975832 0.98339  0.985774 0.979001 0.982439 0.980617] ClfSpacy(batch=300, dropout=0.5, epochs=20, epochs2=2, frozen=True, lr=0.001, lstm_type=7, max_length=75, n_hidden=512, m=MEAN)
auc=0.9789   2: [0.976508 0.988912 0.986447 0.961075 0.978095 0.982544] ClfSpacy(batch=300, dropout=0.5, epochs=20, epochs2=2, frozen=True, lr=0.002, lstm_type=9, max_length=100, n_hidden=512, m=LINEAR)
auc=0.9773   3: [0.971795 0.982235 0.97305  0.983214 0.97729  0.976365] ClfSpacy(batch=300, dropout=0.3, epochs=20, epochs2=2, frozen=True, lr=0.001, lstm_type=6, max_length=100, n_hidden=512, m=LINEAR)

instance5/spacy_lstm20s.ALL.LINEAR2.csv
Your submission scored 0.9723, which is not an improvement of your best score. Keep trying!

gpu3/spacy_lstm21_flip.40000.log
RESULTS SUMMARY: 20
auc=0.9776   6: get_clf23 ClfSpacy(bat=300, dropout=0.3, lr=0.001, typ=6, max_length=100, n_hidden=512, pm=LINEAR2)
auc=0.9775   7: get_clf23 ClfSpacy(bat=300, dropout=0.3, lr=0.001, typ=6, max_length=100, n_hidden=512, pm=LINEAR3)
auc=0.9775   5: get_clf23 ClfSpacy(bat=300, dropout=0.3, lr=0.001, typ=6, max_length=100, n_hidden=512, pm=LINEAR)
auc=0.9771   4: get_clf23 ClfSpacy(bat=300, dropout=0.3, lr=0.001, typ=6, max_length=100, n_hidden=512, pm=MEAN)
auc=0.9765  10: get_clf24 ClfSpacy(bat=300, dropout=0.5, lr=0.002, typ=6, max_length=100, n_hidden=512, pm=LINEAR2)

instance3/spacy_lstm19_flip.20000.log
RESULTS SUMMARY: 32
auc=0.9804  23: get_clf24 ClfSpacy(bat=300, dropout=0.5, lr=0.001, typ=7, max_length=100, n_hidden=512, pm=LINEAR3)
auc=0.9803  22: get_clf24 ClfSpacy(bat=300, dropout=0.5, lr=0.001, typ=7, max_length=100, n_hidden=512, pm=LINEAR2)
auc=0.9803  20: get_clf24 ClfSpacy(bat=300, dropout=0.5, lr=0.001, typ=7, max_length=100, n_hidden=512, pm=MEAN)
auc=0.9801  21: get_clf24 ClfSpacy(bat=300, dropout=0.5, lr=0.001, typ=7, max_length=100, n_hidden=512, pm=LINEAR)
auc=0.9790   0: get_clf23 ClfSpacy(bat=150, dropout=0.3, lr=0.001, typ=9, max_length=100, n_hidden=512, pm=MEAN)
auc=0.9789   3: get_clf23 ClfSpacy(bat=150, dropout=0.3, lr=0.001, typ=9, max_length=100, n_hidden=512, pm=LINEAR3)


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
   0: auc=0.982 [0.9750561  0.98830847 0.98900364 0.97884294 0.98374052 0.9796811 ] (200, 0.1, 30000, (0.001, 0.001, 0.002, 0.003, 0.0), 150, 1, '6B', 300) ClfLstmGlove(bat=64, dropout=0.1, embed_name=6B, embed_size=300, epochs=40, learning_rate=(0.001, 0.001, 0.002, 0.003, 0.0), max_features=30000, maxlen=150, n_folds=1, n_hidden=200)
   1: auc=0.981 [0.97351159 0.98882149 0.98829532 0.97717715 0.98304252 0.97756941] (200, 0.1, 30000, (0.001, 0.001, 0.002, 0.003, 0.0), 150, 1, '840B', 300) ClfLstmGlove(bat=64, dropout=0.1, embed_name=840B, embed_size=300, epochs=40, learning_rate=(0.001, 0.001, 0.002, 0.003, 0.0), max_features=30000, maxlen=150, n_folds=1, n_hidden=200)
   2: auc=0.981 [0.96983904 0.98820016 0.98936429 0.97642246 0.98377134 0.97741031] (200, 0.1, 30000, (0.001, 0.001, 0.002, 0.003, 0.0), 150, 2, '840B', 300) ClfLstmGlove(bat=64, dropout=0.1, embed_name=840B, embed_size=300, epochs=40, learning_rate=(0.001, 0.001, 0.002, 0.003, 0.0), max_features=30000, maxlen=150, n_folds=1, n_hidden=200)
   3: auc=0.980 [0.97371822 0.98834821 0.98607619 0.97394626 0.98195597 0.97681372] (200, 0.1, 20000, (0.001, 0.001, 0.002, 0.003, 0.0), 150, 1, '840B', 300) ClfLstmGlove(bat=64, dropout=0.1, embed_name=840B, embed_size=300, epochs=40, learning_rate=(0.001, 0.001, 0.002, 0.003, 0.0), max_features=20000, maxlen=150, n_folds=1, n_hidden=200)
   4: auc=0.980 [0.9757477  0.98809623 0.98858001 0.97236212 0.98183801 0.97235197] (200, 0.1, 30000, (0.001, 0.001, 0.002, 0.003, 0.0), 200, 1, '840B', 300) ClfLstmGlove(bat=64, dropout=0.1, embed_name=840B, embed_size=300, epochs=40, learning_rate=(0.001, 0.001, 0.002, 0.003, 0.0), max_features=30000, maxlen=200, n_folds=1, n_hidden=200)
   5: auc=0.979 [0.97508761 0.98684306 0.9888123  0.96740227 0.98302377 0.97569144] (200, 0.2, 30000, (0.001, 0.001, 0.002, 0.003, 0.0), 150, 1, '840B', 300) ClfLstmGlove(bat=64, dropout=0.2, embed_name=840B, embed_size=300, epochs=40, learning_rate=(0.001, 0.001, 0.002, 0.003, 0.0), max_features=30000, maxlen=150, n_folds=1, n_hidden=200)
   6: auc=0.979 [0.97140636 0.98771753 0.98595271 0.97324315 0.98139698 0.97699946] (200, 0.1, 30000, (0.001, 0.001, 0.002, 0.003, 0.0), 150, 1, '6B', 200) ClfLstmGlove(bat=64, dropout=0.1, embed_name=6B, embed_size=200, epochs=40, learning_rate=(0.001, 0.001, 0.002, 0.003, 0.0), max_features=30000, maxlen=150, n_folds=1, n_hidden=200)
   7: auc=0.979 [0.97363074 0.98655193 0.98809387 0.97010162 0.98237335 0.97454368] (200, 0.1, 30000, (0.002, 0.002, 0.002, 0.003, 0.0), 150, 1, '840B', 300) ClfLstmGlove(bat=64, dropout=0.1, embed_name=840B, embed_size=300, epochs=40, learning_rate=(0.002, 0.002, 0.002, 0.003, 0.0), max_features=30000, maxlen=150, n_folds=1, n_hidden=200)
   8: auc=0.978 [0.96968342 0.98657204 0.98447823 0.97598084 0.97960141 0.97371561] (200, 0.1, 30000, (0.001, 0.001, 0.002, 0.003, 0.0), 70, 1, '840B', 300) ClfLstmGlove(bat=64, dropout=0.1, embed_name=840B, embed_size=300, epochs=40, learning_rate=(0.001, 0.001, 0.002, 0.003, 0.0), max_features=30000, maxlen=70, n_folds=1, n_hidden=200)
   9: auc=0.978 [0.97274651 0.98564288 0.98787301 0.96730656 0.98123012 0.97244824] (200, 0.1, 30000, (0.001, 0.001, 0.002, 0.003, 0.0), 150, 3, '840B', 300) ClfLstmGlove(bat=64, dropout=0.1, embed_name=840B, embed_size=300, epochs=40, learning_rate=(0.001, 0.001, 0.002, 0.003, 0.0), max_features=30000, maxlen=150, n_folds=1, n_hidden=200)

instance 3
   0: auc=0.981 [0.97338312 0.9891582  0.98879899 0.97496824 0.98250596 0.97656324] (200, 0.1, 30000, (0.002, 0.002, 0.002, 0.003, 0.0), 150, 1, '840B', 300) ClfLstmGlove(bat=64, dropout=0.1, embed_name=840B, embed_size=300, epochs=40, learning_rate=(0.002, 0.002, 0.002, 0.003, 0.0), max_features=30000, maxlen=150, n_folds=1, n_hidden=200)
   1: auc=0.981 [0.97641332 0.98821867 0.98859288 0.97485211 0.98316937 0.97368016] (200, 0.1, 30000, (0.002, 0.002, 0.002, 0.003, 0.0), 150, 2, '840B', 300) ClfLstmGlove(bat=64, dropout=0.1, embed_name=840B, embed_size=300, epochs=40, learning_rate=(0.002, 0.002, 0.002, 0.003, 0.0), max_features=30000, maxlen=150, n_folds=1, n_hidden=200)
   2: auc=0.981 [0.9737291  0.98868542 0.98822773 0.97557122 0.98238556 0.97572784] (200, 0.1, 30000, (0.001, 0.001, 0.002, 0.003, 0.0), 150, 1, '840B', 300) ClfLstmGlove(bat=64, dropout=0.1, embed_name=840B, embed_size=300, epochs=40, learning_rate=(0.001, 0.001, 0.002, 0.003, 0.0), max_features=30000, maxlen=150, n_folds=1, n_hidden=200)
   3: auc=0.979 [0.9753885  0.98834315 0.98859767 0.96284708 0.98333834 0.97814435] (200, 0.2, 30000, (0.001, 0.001, 0.002, 0.003, 0.0), 150, 1, '840B', 300) ClfLstmGlove(bat=64, dropout=0.2, embed_name=840B, embed_size=300, epochs=40, learning_rate=(0.001, 0.001, 0.002, 0.003, 0.0), max_features=30000, maxlen=150, n_folds=1, n_hidden=200)
   4: auc=0.979 [0.97369484 0.98702978 0.9896832  0.96533211 0.98305046 0.97592929] (200, 0.1, 30000, (0.002, 0.002, 0.002, 0.003, 0.0), 150, 3, '840B', 300) ClfLstmGlove(bat=64, dropout=0.1, embed_name=840B, embed_size=300, epochs=40, learning_rate=(0.002, 0.002, 0.002, 0.003, 0.0), max_features=30000, maxlen=150, n_folds=1, n_hidden=200)
   5: auc=0.978 [0.97171631 0.98800955 0.98651588 0.96530701 0.9813796  0.97763309] (200, 0.1, 30000, (0.002, 0.002, 0.002, 0.003, 0.0), 200, 1, '840B', 300) ClfLstmGlove(bat=64, dropout=0.1, embed_name=840B, embed_size=300, epochs=40, learning_rate=(0.002, 0.002, 0.002, 0.003, 0.0), max_features=30000, maxlen=200, n_folds=1, n_hidden=200)
