# coding: utf-8
"""
    There are two very different strong baselines currently in the kernels for this competition:

- An *LSTM* model, which uses a recurrent neural network to model state across each text, with no
  feature engineering
- An *NB-SVM* inspired model, which uses a simple linear approach on top of naive bayes features

 In theory, an ensemble works best when the individual models are as different as possible.
  Therefore, we should see that even a simple average of these two models gets a good result.
  Let's try it! First, we'll load the outputs of the models (in the Kaggle Kernels environment
  you can add these as input files directly from the UI; otherwise you'll need to download them first).
"""
import pandas as pd
from os.path import join
from utils import dim
from framework import SUBMISSION_DIR, LABEL_COLS
from gru_framework import SUBMISSION_DIR as SUBMISSION_DIR2


lstm_path = join(SUBMISSION_DIR2, 'submission_gr_fasttext5.csv')
nb_path = join(SUBMISSION_DIR, 'tfidf_nb.csv')
ensemble_path = join(SUBMISSION_DIR, 'ensemble_nb_fasttext5.csv')

lstm = pd.read_csv(lstm_path)
nb = pd.read_csv(nb_path)

# Now we can take the average of the label columns.
ensemble = lstm.copy()
ensemble[LABEL_COLS] = (nb[LABEL_COLS] + lstm[LABEL_COLS]) / 2

print('lstm:    ', dim(lstm))
print('nb:      ', dim(nb))
print('ensemble:', dim(ensemble))

for i in 0, 1:
    assert lstm.shape[i] == nb.shape[i]
    assert lstm.shape[i] == ensemble.shape[i]

# And finally, create our CSV.
ensemble.to_csv(ensemble_path, index=False)
print('Saved to %s' % ensemble_path)
print('=' * 80)

"""
    ClfLstmGlove(batch_size=64, dropout=0.1, embed_name=840B, embed_size=300, epochs=40,
    learning_rate=[0.002, 0.003, 0.0], max_features=20000, maxlen=100)

    Your submission scored 0.9806, which is not an improvement of your best score. Keep trying!
    ============================================================================

    lstm_path = join(SUBMISSION_DIR, 'lstm_glove.csv')
    nb_path = join(SUBMISSION_DIR, 'tfidf_nb.csv')
    ensemble_path = join(SUBMISSION_DIR, 'ensemble_nb_lstm.csv')

    Your Best Entry
    You advanced 444 places on the leaderboard! (661 out of 2017: 33%)
    Your submission scored 0.9808, which is an improvement of your previous score of 0.9772. Great job!
    Tweet this!
    -------------------------------------------------------

    lstm_path = join(SUBMISSION_DIR, 'lstm_glove_ 50_100_20000_0.100.csv')
    nb_path = join(SUBMISSION_DIR, 'tfidf_nb.csv')
    ensemble_path = join(SUBMISSION_DIR, 'ensemble_nb_lstm_glove_50_100_20000_0.100.csv')

    Your submission scored 0.9807, which is not an improvement of your best score. Keep trying
"""
