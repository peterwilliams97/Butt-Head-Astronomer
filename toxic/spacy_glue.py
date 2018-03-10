# coding: utf-8
"""
"""
import numpy as np
import os
from collections import defaultdict
import spacy
from spacy.language import Language
from utils import (xprint, save_json, load_json, save_pickle, load_pickle, save_pickle_gzip,
    load_pickle_gzip)


SPACY_VECTOR_SIZE = 300  # To match SpaCY vectors
SPACY_VECTOR_TYPE = np.float32
SPACY_ROOT = 'spacy.cache'
GLOVE50 = 'glove.6B.50'

EMBEDDING_TYPE = GLOVE50


if EMBEDDING_TYPE == GLOVE50:
    VECTORS_PATH = os.path.expanduser('~/data/glove.6B/glove.6B.50d.txt')
    assert os.path.exists(VECTORS_PATH), VECTORS_PATH
    VECTORS_DIM = 50
    SPACY_DIR = os.path.join(SPACY_ROOT, GLOVE50)
else:
    SPACY_DIR = SPACY_ROOT
    VECTORS_PATH = None


class SpacyCache:

    def __init__(self):
        os.makedirs(SPACY_DIR, exist_ok=True)
        self.text_tokens_path = os.path.join(SPACY_DIR, 'text.tokens.json')
        self.token_vector_path = os.path.join(SPACY_DIR, 'token.vector.pkl')
        self.text_tokens = load_json(self.text_tokens_path, {})
        self.token_vector = load_pickle(self.token_vector_path, {})
        self.text_tokens_len = len(self.text_tokens)
        self.token_vector_len = len(self.token_vector)
        self.nlp = spacy.load('en_core_web_lg')
        self.n_calls = 0

    def _save(self, min_delta=0):
        if self.text_tokens_len + min_delta < len(self.text_tokens):
            print('_save 1: %7d = %7d + %4d %s' % (len(self.text_tokens),
                self.text_tokens_len, len(self.text_tokens) - self.text_tokens_len,
                self.text_tokens_path))
            save_json(self.text_tokens_path, self.text_tokens)
            self.text_tokens_len = len(self.text_tokens)
        if self.token_vector_len + 2 * min_delta < len(self.token_vector):
            print('_save 2: %7d = %7d + %4d %s' % (len(self.token_vector),
                self.token_vector_len, len(self.token_vector) - self.token_vector_len,
                self.token_vector_path))
            save_pickle(self.token_vector_path, self.token_vector)
            self.token_vector_len = len(self.token_vector)

    def tokenize(self, text):
        """Use SpaCy tokenization and word vectors"""
        tokens = self.text_tokens.get(text)
        if tokens is not None:
            return tokens, 0
        doc = self.nlp(text)
        tokens = []
        for t in doc:
            tokens.append(t.text)
            self.token_vector[t.text] = t.vector
        self.text_tokens[text] = tokens
        if self.n_calls % 1000 == 1:
            print('** %d' % self.n_calls)
            self._save()
        self.n_calls += 1

        return tokens, 1

    def flush(self):
        self._save()


def load_foreign_embeddings():
    local_path = os.path.join(SPACY_DIR, 'nlp.embeddings.gzip')

    if not os.path.exists(local_path):
        # start off with a blank Language class
        print('load_foreign_embeddings: loading original')
        nlp = Language()
        with open(VECTORS_PATH, 'rb') as f:
            # header = f.readline()
            # nr_row, nr_dim = header.split()
            # nlp.vocab.clear_vectors(n_dim)
            for i, line in enumerate(f):
                if i % 50000 == 1000:
                    print('^^^', i)
                line = line.decode('utf8')
                # pieces = line.split()
                # word = pieces[0]
                # vector = np.asarray([float(v) for v in pieces[1:]], dtype='f')

                parts = line.strip().split()
                word = parts[0]
                vector = np.asarray(parts[1:], dtype='float32')

                nlp.vocab.set_vector(word, vector)  # add the vectors to the vocab

        #     embeddings_index = {}
        # with open(embeddings_path, 'rb') as f:
        #     t0 = time.clock()
        #     for i, line in enumerate(f):
        #         parts = line.strip().split()
        #         embeddings_index[parts[0]] = np.asarray(parts[1:], dtype='float32')
        #         if (i + 1) % 200000 == 0:
        #             print('%7d embeddings %4.1f sec' % (i + 1, time.clock() - t0))
        # xprint('%7d embeddings %4.1f sec' % (len(embeddings_index), time.clock() - t0))

        save_pickle_gzip(local_path, nlp)

    nlp = load_pickle_gzip(local_path)

    # test the vectors and similarity
    text = 'class colspan'
    doc = nlp(text)
    print('load_foreign_embeddings:', text, doc[0].similarity(doc[1]))
    return nlp


class SpacySentenceWordCache:

    def __init__(self):
        os.makedirs(SPACY_DIR, exist_ok=True)
        self.text_sents_path = os.path.join(SPACY_DIR, 'sentence.text.tokens.gzip')
        self.text_token_count_path = os.path.join(SPACY_DIR, 'sentence.text.tokens.count.gzip')
        self.text_sents = load_pickle_gzip(self.text_sents_path, {})
        self.text_token_count = load_pickle_gzip(self.text_token_count_path, {})
        xprint("SpacySentenceWordCache: sent path=%s len=%d" % (self.text_sents_path, len(self.text_sents)))
        self.text_sents_len = len(self.text_sents)
        self.text_token_count_len = self._total_counts()
        self.nlp = None
        self.nlp = self._load_nlp()
        self.n_calls = 0

    def sentence_lengths(self):
        n_docs = len(self.text_sents)
        n_sents = sum(len(sent) for sent in self.text_sents.values())
        lens = []
        for sents in self.text_sents.values():
            lens.extend(len(v) for v in sents)
        assert len(lens) == n_sents, (len(lens), n_sents)
        thresholds = [(t, len([n for n in lens if n <= t])) for t in (25, 50, 75, 100, 500, 5000)]
        lens = np.array(lens)
        xprint('n_docs=%d' % n_docs)
        xprint('n_sents=%d' % n_sents)
        xprint('sents per doc=%.1f' % (n_sents / n_docs))
        xprint('sentence lengths: min mean max= %.1f %.1f %.1f' % (np.min(lens), np.mean(lens),
            np.max(lens)))
        xprint('thresholds')
        for t, n in thresholds:
            xprint('%6d: %8d %.3f' % (t, n, n / n_sents))

    def _total_counts(self):
        return sum(sum(v.values()) for v in self.text_token_count.values())

    def _load_nlp(self):
        if self.nlp is None:
            print("Loading SpacySentenceWordCache")
            # nlp = spacy.load('en_vectors_web_lg')
            nlp = load_foreign_embeddings()
            nlp.add_pipe(nlp.create_pipe('tagger'))
            nlp.add_pipe(nlp.create_pipe('parser'))
            nlp.add_pipe(nlp.create_pipe('sentencizer'))
            self.nlp = nlp
        return self.nlp

    def _save(self, min_delta=0):
        if self.text_sents_len + min_delta < len(self.text_sents):
            print('_save 1: %7d = %7d + %4d %s' % (len(self.text_sents),
                self.text_sents_len, len(self.text_sents) - self.text_sents_len,
                self.text_sents_path))
            save_pickle_gzip(self.text_sents_path, self.text_sents)
            self.text_sents_len = len(self.text_sents)

        if self.text_token_count_len + min_delta < self._total_counts():
            print('_save 2: %7d = %7d + %4d %s' % (len(self.text_token_count),
                self.text_sents_len, self._total_counts() - self.text_token_count_len,
                self.text_token_count_path))
            save_pickle_gzip(self.text_token_count_path, self.text_token_count)
            self.text_token_count_len = self._total_counts()

    def sent_id_pipe(self, texts_in):
        """Use SpaCy tokenization and word vectors"""
        loaded = set(self.text_sents) & set(self.text_token_count)
        texts = [text for text in texts_in if text not in loaded]
        # texts = [text for text in texts_in]
        if texts:
            nlp = self._load_nlp()
            for text, doc in zip(texts, nlp.pipe(texts)):
                self.text_sents[text] = []
                token_count = defaultdict(int)
                for sent in doc.sents:
                    sent_ids = []
                    for token in sent:
                        vector_id = token.vocab.vectors.find(key=token.orth)
                        # print('@@@ %-20r %d' % (token.text, vector_id))
                        sent_ids.append(vector_id)
                        token_count[vector_id] += 1
                    self.text_sents[text].append(sent_ids)
                self.text_token_count[text] = token_count

                if self.n_calls % 10000 == 1:
                    print('**sent_id_pipe: n_calls=%d' % self.n_calls)
                    self._save()
                self.n_calls += 1

        word_count = defaultdict(int)
        for text in texts_in:
            for w, c in self.text_token_count[text].items():
                word_count[w] += c

        return [self.text_sents[text] for text in texts_in], word_count

    def flush(self):
        self._save()


class SpacySentenceCharCache:

    def __init__(self):
        os.makedirs(SPACY_DIR, exist_ok=True)
        self.text_sent_chars_path = os.path.join(SPACY_DIR, 'sentence.text.texts.gzip')
        self.text_sent_chars = load_pickle_gzip(self.text_sent_chars_path, {})
        print("SpacySentenceCache: text path=%s len=%d" % (self.text_sent_chars_path, len(self.text_sent_chars)))
        self.text_sent_chars_len = len(self.text_sent_chars)

        self.nlp = None
        self.n_calls = 0

    def _load_nlp(self):
        if self.nlp is None:
            print("Loading SpacySentenceCharCache")
            nlp = spacy.load('en_core_web_lg')
            nlp.add_pipe(nlp.create_pipe('sentencizer'))
            self.nlp = nlp
        return self.nlp

    def _save(self, min_delta=0):
        if self.text_sent_chars_len + min_delta < len(self.text_sent_chars):
            print('_save 2: %7d = %7d + %4d %s' % (len(self.text_sent_chars),
                self.text_sent_chars_len, len(self.text_sent_chars) - self.text_sent_chars_len,
                self.text_sent_chars_path))
            save_pickle_gzip(self.text_sent_chars_path, self.text_sent_chars)
            self.text_sent_chars_len = len(self.text_sent_chars)

    def sent_char_pipe(self, char_index, texts_in):
        """Look up char indexes for chars in sentences in each text in `texts_in`
            char_index: Index dict for character embeddings
            texts_in: Texts to tokenize into sentences
            Returns: [sent_chars[text] for text in texts_in] where
                    sent_chars[text] is a list of list of character indexes, one list for each
                    sentence
        """
        texts = [text for text in texts_in if text not in self.text_sent_chars]
        if texts:
            nlp = self._load_nlp()
            for text, doc in zip(texts, nlp.pipe(texts)):
                self.text_sent_chars[text] = [[char_index.get(c, -1) for c in sent.string.strip()]
                                              for sent in doc.sents]
                if self.n_calls % 10000 == 1:
                    print('**sent_char_pipe: n_calls=%d' % self.n_calls)
                    self._save()
                self.n_calls += 1

        return [self.text_sent_chars[text] for text in texts_in]

    def flush(self):
        self._save()


if __name__ == '__main__':
    sentence_cache = SpacySentenceWordCache()
    sentence_cache.sentence_lengths()
    nlp = sentence_cache._load_nlp()
    nlp.pipeline = [
        ('tagger', nlp.tagger),
        ('parser', nlp.parser),
    ]
    print('+++++ pipe_names=%s' % nlp.pipe_names)

    texts = ['I am number %d' % i for i in range(10)]
    for i, doc in enumerate(nlp.pipe(texts, batch_size=1)):
        print(i, doc)
