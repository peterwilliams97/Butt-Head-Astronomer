# coding: utf-8
"""
"""
import numpy as np
import os
from collections import defaultdict
import re
import spacy
from utils import xprint, save_json, save_pickle_gzip, load_pickle_gzip


SPACY_VECTOR_SIZE = 300  # To match SpaCY vectors
SPACY_VECTOR_TYPE = np.float32
SPACY_DIR = 'spacy.sentence.tokens'
RE_SPACE = re.compile(r'\s+', re.DOTALL | re.MULTILINE)


def islowercase(w):
    if not w.isalpha():
        return True
    if not all(ord(c) < 128 for c in w):
        return True
    return w.islower()


class SpacySentenceTokenizer:

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
        self.nlp = self._load_nlp()  # !@#$
        self.n_calls = 0

    def show_sentence_lengths(self):
        n_docs = len(self.text_sents)
        n_sents = sum(len(sent) for sent in self.text_sents.values())
        lens = []
        for sents in self.text_sents.values():
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
        return sum(sum(v.values()) for v in self.text_token_count.values())

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

    def sentence_tokens(self, texts_in, lowercase):
        """Use SpaCy tokenization """
        assert lowercase
        loaded = set(self.text_sents) & set(self.text_token_count)
        texts = [text for text in texts_in if text not in loaded]
        # texts = [text for text in texts_in]
        if texts:
            nlp = self._load_nlp()
            for text, doc in zip(texts, nlp.pipe(texts)):
                self.text_sents[text] = []
                token_count = defaultdict(int)
                for sent in doc.sents:
                    words = [token.text for token in sent if not RE_SPACE.search(token.text)]
                    self.text_sents[text].append(words)
                    for w in words:
                        token_count[w] += 1
                self.text_token_count[text] = token_count

                if self.n_calls % 10000 == 1:
                    print('**sentence_tokens: n_calls=%d' % self.n_calls)
                    self._save()
                self.n_calls += 1

        if lowercase:
            for text in texts_in:
                self.text_sents[text] = [[w.lower() for w in sent]
                                         for sent in self.text_sents[text]]
                # for sent in self.text_sents[text]:
                #     for w in sent:
                #         assert islowercase(w), (w, sent, text[:200])

            # for text in texts_in:
            #     for sent in self.text_sents[text]:
            #         for w in sent:
            #             assert islowercase(w), (w, sent, text[:200])

        word_count = defaultdict(int)
        for text in texts_in:
            for w, c in self.text_token_count[text].items():
                if lowercase:
                    w = w.lower()
                word_count[w] += c
                # assert islowercase(w), (w)

        # print('!!!', [self.text_sents[text] for text in texts_in[:2]])
        return [self.text_sents[text] for text in texts_in], word_count

    def flush(self):
        self._save()


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
    text_sents, word_count = sentence_cache.sentence_tokens(texts)
    sentence_cache.show_sentence_lengths()
