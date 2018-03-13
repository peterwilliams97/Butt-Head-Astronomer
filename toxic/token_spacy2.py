# coding: utf-8
"""
"""
import numpy as np
import os
from collections import defaultdict
import re
import spacy
from utils import xprint, save_pickle, load_pickle


SPACY_VECTOR_SIZE = 300  # To match SpaCY vectors
SPACY_VECTOR_TYPE = np.float32
SPACY_DIR_ = 'spacy.sentence.tokens'
RE_SPACE = re.compile(r'^\s+$', re.DOTALL | re.MULTILINE)


def islowercase(w):
    if not w.isalpha():
        return True
    if not all(ord(c) < 128 for c in w):
        return True
    return w.islower()


class SpacySentenceTokenizer:

    def __init__(self, method=1):
        self.method = method
        spacy_dir = '%s.%d' % (SPACY_DIR_, method)
        os.makedirs(spacy_dir, exist_ok=True)
        self.sent_texts_path = os.path.join(spacy_dir, 'sentence.text.tokens.pkl')
        self.token_count_path = os.path.join(spacy_dir, 'sentence.text.tokens.count.pkl')
        self.sent_texts = load_pickle(self.sent_texts_path, {})
        self.token_count = load_pickle(self.token_count_path, {})
        xprint("SpacySentenceWordCache: sent path=%s len=%d" % (self.sent_texts_path, len(self.sent_texts)))
        self.sent_texts_len = len(self.sent_texts)
        self.token_count_len = self._total_counts()
        self.nlp = None
        # self.nlp = self._load_nlp()  # !@#$
        self.n_calls = 0

    def show_sentence_lengths(self):
        n_docs = len(self.sent_texts)
        n_sents = sum(len(sent) for sent in self.sent_texts.values())
        lens = []
        for sents in self.sent_texts.values():
            lens.extend(len(v) for v in sents)
        assert len(lens) == n_sents, (len(lens), n_sents)
        thresholds = [(t, len([n for n in lens if n <= t])) for t in (25, 50, 75, 100, 500, 5000)]
        lens = np.array(lens)
        xprint('n_docs=%d' % n_docs)
        if n_docs == 0:
            xprint('No documents processed')
            return
        xprint('n_sents=%d' % n_sents)
        xprint('sents per doc=%.1f' % (n_sents / n_docs))
        xprint('sentence lengths: min mean max= %.1f %.1f %.1f' % (np.min(lens), np.mean(lens),
            np.max(lens)))
        xprint('thresholds')
        for t, n in thresholds:
            xprint('%6d: %8d %.3f' % (t, n, n / n_sents))

    def _total_counts(self):
        return sum(sum(v.values()) for v in self.token_count.values())

    def _load_nlp(self):
        if self.nlp is None:
            model = 'en_core_web_lg'
            # model = 'en'
            print("Loading SpacySentenceWordCache: %s" % model)
            nlp = spacy.load(model)
            nlp.add_pipe(nlp.create_pipe('sentencizer'))
            self.nlp = nlp
        return self.nlp

    def _save(self, min_delta=0):
        if self.sent_texts_len + min_delta < len(self.sent_texts):
            print('_save 1: %7d = %7d + %4d %s' % (len(self.sent_texts),
                self.sent_texts_len, len(self.sent_texts) - self.sent_texts_len,
                self.sent_texts_path))
            save_pickle(self.sent_texts_path, self.sent_texts)
            self.sent_texts_len = len(self.sent_texts)

        if self.token_count_len + min_delta < self._total_counts():
            print('_save 2: %7d = %7d + %4d %s' % (len(self.token_count),
                self.token_count_len, self._total_counts() - self.token_count_len,
                self.token_count_path))
            save_pickle(self.token_count_path, self.token_count)
            self.token_count_len = self._total_counts()

    def token_lists(self, texts_in, max_length):
        """Use SpaCy tokenization """
        assert isinstance(max_length, int), max_length
        loaded = set(self.sent_texts) & set(self.token_count)
        texts = [text for text in texts_in if text not in loaded]
        stride = (max_length + 1) // 2
        # texts = [text for text in texts_in]
        if texts:
            nlp = self._load_nlp()
            vectors = nlp.vocab.vectors
            lowered = defaultdict(int)

            if self.method == 0:
                for text, doc in zip(texts, nlp.pipe(texts)):
                    tokens = [token for token in doc if not RE_SPACE.search(token.text)]
                    sentences = [tokens[i:i + max_length] for i in range(0, len(tokens), stride)]
                    self.sent_texts[text] = [[t.text for t in sent] for sent in sentences]
                    self.token_count[text] = get_token_count(tokens)

                    if self.n_calls % 10000 == 1:
                        print('**token_lists: n_calls=%d' % self.n_calls)
                        self._save()
                    self.n_calls += 1
            elif self.method == 1:
                for text, doc in zip(texts, nlp.pipe(texts)):
                    tokens = []
                    for sent in doc.sents:
                        toks = [token.text for token in doc if not RE_SPACE.search(token.text)]
                        # print(' toks= %d %s' % (len(toks), toks))
                        tokens.extend(toks)
                    # print(' tokens=%d %s' % (len(tokens), tokens))

                    for i, w in enumerate(tokens):
                        if vectors.find(key=w) < 1:
                            w_l = w.lower()
                            if vectors.find(key=w_l) >= 0:
                                tokens[i] = w_l
                                lowered[w] += 1

                    sentences = []
                    i = 0
                    while True:
                        sentences.append(tokens[i:i + max_length])
                        delta = 10 if i > 0 else 0
                        if i + max_length + delta >= len(tokens):
                            break
                        i += stride

                    # sentences = [tokens[i:i + max_length] for i in range(0, len(tokens), stride)]

                    self.sent_texts[text] = sentences # [[t.text for t in sent] for sent in sentences]
                    self.token_count[text] = get_token_count(tokens)
                    # print('sentences=%s' % ['%d: %s\n' % (i, s) for i, s in enumerate(sentences)])
                    # assert False

                    if self.n_calls % 10000 == 1:
                        print('**token_lists: n_calls=%d' % self.n_calls)
                        self._save()
                    self.n_calls += 1

        word_count = defaultdict(int)
        for text in texts_in:
            for w, c in self.token_count[text].items():
                word_count[w] += c

        print('lowered=%d %d %s' % (len(lowered), sum(lowered.values()), sorted(lowered)[:20]))

        # print('!!!', [self.sent_texts[text] for text in texts_in[:2]])
        return [self.sent_texts[text] for text in texts_in], word_count

    def flush(self):
        self._save()


def get_token_count(tokens):
    token_count = defaultdict(int)
    for w in tokens:
        token_count[w] += 1
    return token_count


if __name__ == '__main__':
    sentence_cache = SpacySentenceTokenizer()
    sentence_cache.show_sentence_lengths()
    nlp = sentence_cache._load_nlp()
    print('+++++ pipe_names=%s' % nlp.pipe_names)

    sents = ['I am number %d %s %s. ' % (i, 'yes ' * (i % 10), chr(i + 32)) for i in range(100)]
    for n in (10, 20, 50, 100):
        sents.append(('My name is John %d and ' % n) * n)
    texts = [' '.join(sents[:(i % len(sents)) + 1]) for i in range(2 * len(sents))]
    # for n in (100, 200, 500, 1000):
    #     texts.append('!'.join(texts[:n]))
    sent_texts, word_count = sentence_cache.token_lists(texts)
    sentence_cache.show_sentence_lengths()
