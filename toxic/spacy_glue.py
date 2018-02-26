# coding: utf-8
"""
"""
import numpy as np
import os
import spacy
from utils import save_json, load_json, save_pickle, load_pickle, save_pickle_gzip, load_pickle_gzip


SPACY_VECTOR_SIZE = 300  # To match SpaCY vectors
SPACY_VECTOR_TYPE = np.float32
SPACY_DIR = 'spacy.cache'


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


class SpacySentenceCache:

    def __init__(self):
        os.makedirs(SPACY_DIR, exist_ok=True)
        self.text_sents_path = os.path.join(SPACY_DIR, 'sentence.text.tokens.gzip')
        self.text_sents = load_pickle_gzip(self.text_sents_path, {})
        print("SpacySentenceCache: sent path=%s len=%d" % (self.text_sents_path, len(self.text_sents)))
        self.text_sents_len = len(self.text_sents)

        self.text_sent_chars_path = os.path.join(SPACY_DIR, 'sentence.text.texts.gzip')
        self.text_sent_chars = load_pickle_gzip(self.text_sent_chars_path, {})
        print("SpacySentenceCache: text path=%s len=%d" % (self.text_sent_chars_path, len(self.text_sent_chars)))
        self.text_sent_chars_len = len(self.text_sent_chars)

        self.nlp = None
        self.n_calls = 0

    def _load_nlp(self):
        if self.nlp is None:
            print("Loading SpacySentenceCache")
            nlp = spacy.load('en_core_web_lg')
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
        if self.text_sent_chars_len + min_delta < len(self.text_sent_chars):
            print('_save 2: %7d = %7d + %4d %s' % (len(self.text_sent_chars),
                self.text_sent_chars_len, len(self.text_sent_chars) - self.text_sent_chars_len,
                self.text_sent_chars_path))
            save_pickle_gzip(self.text_sent_chars_path, self.text_sent_chars)
            self.text_sent_chars_len = len(self.text_sent_chars)

    def sent_id_pipe(self, texts_in):
        """Use SpaCy tokenization and word vectors"""
        texts = [text for text in texts_in if text not in self.text_sents]
        if texts:
            nlp = self._load_nlp()
            for text, doc in zip(texts, nlp.pipe(texts)):
                self.text_sents[text] = []
                for sent in doc.sents:
                    sent_ids = []
                    for token in sent:
                        vector_id = token.vocab.vectors.find(key=token.orth)
                        sent_ids.append(vector_id)
                    self.text_sents[text].append(sent_ids)

                if self.n_calls % 10000 == 1:
                    print('**sent_id_pipe: n_calls=%d' % self.n_calls)
                    self._save()
                self.n_calls += 1

        return [self.text_sents[text] for text in texts_in]

    def sent_char_pipe(self, char_index, texts_in):
        """Use SpaCy tokenization"""
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


# class SpacyFeatureCache:

#     def __init__(self):
#         os.makedirs(SPACY_DIR, exist_ok=True)
#         self.sents_path = os.path.join(SPACY_DIR, 'sentence.features.gzip')
#         self.sents = load_pickle_gzip(self.sents_path, {})
#         print("SpacyFeatureCache: path=%s len=%d" % (self.sents_path, len(self.sents)))
#         self.sents_len = len(self.sents)
#         self.n_calls = 0

#     def _save(self, min_delta=0):
#         if self.sents_len + min_delta < len(self.sents):
#             print('_save 1: %7d = %7d + %4d %s' % (len(self.sents),
#                 self.sents_len, len(self.sents) - self.sents_len,
#                 self.ents_path))
#             save_pickle_gzip(self.sents_path, self.sents)
#             self.sents_len = len(self.sents)

#     def sent_id_pipe(self, docs_in):
#         """Use SpaCy tokenization and word vectors"""

#         docs = list(docs_in)
#         doc_ids_list = []
#         for doc in docs:
#             doc_ids = []
#             for token in doc:
#                 doc_ids.append(token.vocab.vectors.find(key=token.orth))
#             doc_ids_list.append(doc_ids)

#         return X
#         texts = [text for text in texts_in if text not in self.text_sents]
#         if texts:
#             nlp = self._load_nlp()
#             for text, doc in zip(texts, nlp.pipe(texts)):
#                 self.text_sents[text] = []
#                 for sent in doc.sents:
#                     sent_ids = []
#                     for token in sent:
#                         vector_id = token.vocab.vectors.find(key=token.orth)
#                         sent_ids.append(vector_id)
#                     self.text_sents[text].append(sent_ids)

#                 if self.n_calls % 10000 == 1:
#                     print('**sent_id_pipe: n_calls=%d' % self.n_calls)
#                     self._save()
#                 self.n_calls += 1

#         return [self.text_sents[text] for text in texts_in]

#     def flush(self):
#         self._save()

