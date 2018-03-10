# coding: utf-8
"""
"""
import spacy



text = '''
    My name is Peter. I am a fäther. Wasn't today fun? John's dog is Molly! $5.93 dxvddf
    rrefeq is a dfsfs.
    who is the ownwe?
'''

for model in ['en_core_web_lg', 'en']:
    print(model, '-' * 80)
    nlp = spacy.load(model)
    # nlp.add_pipe(nlp.create_pipe('tagger'))
    # nlp.add_pipe(nlp.create_pipe('parser'))
    nlp.add_pipe(nlp.create_pipe('sentencizer'))

    doc = nlp(text)
    for sent in doc.sents:
        print('sent=%r' % sent)
        for token in sent:
            vector_id = token.vocab.vectors.find(key=token.orth)
            print('@@@ %-10r, %-10r, %5s, %5s, %5d ' % (
                token.text, token.lemma_,
                token.pos_, token.tag_,
                vector_id))
            assert token.text == token.orth_, (token.text, token.orth_)
