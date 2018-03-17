from keras.models import Sequential
from keras.layers import (LSTM, Dense, Embedding, Bidirectional, Dropout, GlobalMaxPool1D,
    GlobalAveragePooling1D, BatchNormalization, TimeDistributed, Flatten)
from keras.models import Model
from keras.layers import Input, SpatialDropout1D, concatenate
from keras.layers import GRU, GlobalMaxPooling1D
from keras.optimizers import Adam
import math
from utils import xprint, dim


def build_lstm1(embeddings, shape, settings):
    model = Sequential()
    model.add(
        Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            input_length=shape['max_length'],
            trainable=False,
            weights=[embeddings],
            mask_zero=True
        )
    )
    model.add(TimeDistributed(Dense(shape['n_hidden'], use_bias=False)))
    model.add(Bidirectional(LSTM(shape['n_hidden'],
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout'])))
    model.add(Dense(shape['n_class'], activation='sigmoid'))
    xprint('build_lstm1: embeddings=%s shape=%s' % (dim(embeddings), shape))
    return model


def build_lstm2(embeddings, shape, settings):
    # inp = Input(shape=(shape['max_length'],))
    # x = Embedding(
    #         embeddings.shape[0],
    #         embeddings.shape[1],
    #         input_length=shape['max_length'],
    #         trainable=False,
    #         weights=[embeddings],
    #         mask_zero=True
    #     )(inp)
    # x = Bidirectional(LSTM(shape['n_hidden'],
    #                              recurrent_dropout=settings['dropout'],
    #                              dropout=settings['dropout']))(x)
    # x = GlobalMaxPool1D()(x)
    # x = BatchNormalization()(x)
    # x = Dense(50, activation="relu")(x)
    # #x = BatchNormalization()(x)
    # x = Dropout(dropout)(x)
    # x = Dense(shape['n_class'], activation='sigmoid')(x)
    # model = Model(inputs=inp, outputs=x)

    model = Sequential()
    model.add(
        Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            input_length=shape['max_length'],
            trainable=False,
            weights=[embeddings],
            mask_zero=False
        )
    )
    model.add(TimeDistributed(Dense(shape['n_hidden'], use_bias=False), name='td2'))
    model.add(Bidirectional(LSTM(shape['n_hidden'], return_sequences=True,
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout'])))
    model.add(GlobalMaxPool1D())
    model.add(BatchNormalization())
    model.add(Dense(shape['n_class'], activation='sigmoid'))
    xprint('build_lstm2: embeddings=%s shape=%s' % (dim(embeddings), shape))
    return model


def build_lstm3(embeddings, shape, settings):
    model = Sequential()
    model.add(
        Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            input_length=shape['max_length'],
            trainable=False,
            weights=[embeddings],
            mask_zero=False,
            name='eembed'
        )
    )
    model.add(TimeDistributed(Dense(shape['n_hidden'], use_bias=False, name='td3')))
    model.add(Bidirectional(LSTM(shape['n_hidden'], return_sequences=True,
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout'])))
    model.add(Flatten(name='flaaten'))
    model.add(BatchNormalization())
    model.add(Dense(shape['n_class'], activation='sigmoid'))
    xprint('build_lstm3: embeddings=%s shape=%s' % (dim(embeddings), shape))
    return model


def build_lstm4(embeddings, shape, settings):
    model = Sequential()
    model.add(
        Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            input_length=shape['max_length'],
            trainable=False,
            weights=[embeddings],
            mask_zero=False,
            name='eembed'
        )
    )
    model.add(TimeDistributed(Dense(shape['n_hidden'], use_bias=False, name='td4')))
    model.add(Bidirectional(LSTM(shape['n_hidden'], return_sequences=True,
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout'])))
    model.add(Flatten(name='flaaten'))
    model.add(BatchNormalization())
    n_dense = int(math.ceil(math.sqrt(shape['n_hidden'] * shape['n_class'])))
    model.add(Dense(n_dense, activation='relu'))
    # model.add(BatchNormalization())
    # x = Dropout(dropout)(x)
    model.add(Dense(shape['n_class'], activation='sigmoid'))
    xprint('build_lstm4: embeddings=%s shape=%s' % (dim(embeddings), shape))
    return model


def build_lstm5(embeddings, shape, settings):
    model = Sequential()
    model.add(
        Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            input_length=shape['max_length'],
            trainable=False,
            weights=[embeddings],
            mask_zero=False
        )
    )
    model.add(TimeDistributed(Dense(shape['n_hidden'], use_bias=False)))
    model.add(Bidirectional(LSTM(shape['n_hidden'], return_sequences=True,
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout'])))
    model.add(GlobalAveragePooling1D())
    model.add(BatchNormalization())
    model.add(Dense(shape['n_class'], activation='sigmoid'))
    xprint('build_lstm5: embeddings=%s shape=%s' % (dim(embeddings), shape))
    return model


def build_lstm6(embeddings, shape, settings):
    model = Sequential()
    model.add(
        Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            input_length=shape['max_length'],
            trainable=False,
            weights=[embeddings],
            mask_zero=False
        )
    )
    model.add(TimeDistributed(Dense(shape['n_hidden'], use_bias=False), name='td6'))
    model.add(Bidirectional(LSTM(shape['n_hidden'], return_sequences=True,
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout'])))
    model.add(GlobalMaxPool1D())
    model.add(BatchNormalization())
    model.add(Dropout(settings['dropout'] / 2.0))
    model.add(Dense(shape['n_class'], activation='sigmoid'))
    xprint('build_lstm6: embeddings=%s shape=%s' % (dim(embeddings), shape))
    return model


def build_lstm7(embeddings, shape, settings):
    """RELU dense layer
    """
    model = Sequential()
    model.add(
        Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            input_length=shape['max_length'],
            trainable=False,
            weights=[embeddings],
            mask_zero=False
        )
    )
    model.add(TimeDistributed(Dense(shape['n_hidden'], use_bias=False), name='td7'))
    model.add(Bidirectional(LSTM(shape['n_hidden'], return_sequences=True,
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout'])))
    model.add(GlobalMaxPool1D())
    model.add(BatchNormalization())
    # model.add(Dropout(settings['dropout'] / 2.0))
    model.add(Dense(shape['n_hidden'] // 2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(settings['dropout'] / 2.0))
    model.add(Dense(shape['n_class'], activation='sigmoid'))
    xprint('build_lstm7: embeddings=%s shape=%s' % (dim(embeddings), shape))
    return model


def build_lstm8(embeddings, shape, settings):
    """Flatten rather than pool"""
    model = Sequential()
    model.add(
        Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            input_length=shape['max_length'],
            trainable=False,
            weights=[embeddings],
            mask_zero=False,
            name='eembed'
        )
    )
    model.add(TimeDistributed(Dense(shape['n_hidden'], use_bias=False, name='td8')))
    model.add(Bidirectional(LSTM(shape['n_hidden'], return_sequences=True,
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout']), name='bidi'))
    model.add(Flatten(name='flaaten'))
    model.add(BatchNormalization())
    model.add(Dropout(settings['dropout'] / 2.0))
    model.add(Dense(shape['n_class'], activation='sigmoid'))
    xprint('build_lstm8: embeddings=%s shape=%s' % (dim(embeddings), shape))
    return model


def build_lstm9(embeddings, shape, settings):
    """2 layer LSTM
    """
    model = Sequential()
    model.add(
        Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            input_length=shape['max_length'],
            trainable=False,
            weights=[embeddings],
            mask_zero=False
        )
    )
    model.add(TimeDistributed(Dense(shape['n_hidden'], use_bias=False), name='td9a'))
    model.add(Bidirectional(LSTM(shape['n_hidden'], return_sequences=True,
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout']), name='bidi9a'))
    # model.add(GlobalMaxPool1D())
    # model.add(BatchNormalization())
    # model.add(Dropout(settings['dropout'] / 2.0))

    # model.add(TimeDistributed(Dense(shape['n_hidden'], use_bias=False), name='td9b'))
    model.add(Bidirectional(LSTM(shape['n_hidden'], return_sequences=True,
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout']), name='bidi9b'))
    model.add(GlobalMaxPool1D(name='mp9'))
    model.add(BatchNormalization(name='bn9'))
    model.add(Dropout(settings['dropout'] / 2.0, name='drop9b'))

    model.add(Dense(shape['n_class'], activation='sigmoid', name='den9b'))
    xprint('build_lstm9: embeddings=%s shape=%s' % (dim(embeddings), shape))
    return model


def build_lstm10(embeddings, shape, settings):
    """3 layer LSTM
    """
    model = Sequential()
    model.add(
        Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            input_length=shape['max_length'],
            trainable=False,
            weights=[embeddings],
            mask_zero=False
        )
    )
    model.add(TimeDistributed(Dense(shape['n_hidden'], use_bias=False), name='td9a'))
    model.add(Bidirectional(LSTM(shape['n_hidden'], return_sequences=True,
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout']), name='bidi9a'))
    # model.add(GlobalMaxPool1D())
    # model.add(BatchNormalization())
    # model.add(Dropout(settings['dropout'] / 2.0))

    # model.add(TimeDistributed(Dense(shape['n_hidden'], use_bias=False), name='td9b'))
    model.add(Bidirectional(LSTM(shape['n_hidden'], return_sequences=True,
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout']), name='bidi9b'))
    model.add(Bidirectional(LSTM(shape['n_hidden'], return_sequences=True,
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout']), name='bidi9c'))
    model.add(GlobalMaxPool1D(name='mp9'))
    model.add(BatchNormalization(name='bn9'))
    model.add(Dropout(settings['dropout'] / 2.0, name='drop9b'))

    model.add(Dense(shape['n_class'], activation='sigmoid', name='den9b'))
    xprint('build_lstm10: embeddings=%s shape=%s' % (dim(embeddings), shape))
    return model


def build_lstm11(embeddings, shape, settings):
    """3 layer LSTM with fewer weights in higher layers
    """
    model = Sequential()
    model.add(
        Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            input_length=shape['max_length'],
            trainable=False,
            weights=[embeddings],
            mask_zero=False
        )
    )
    model.add(TimeDistributed(Dense(shape['n_hidden'], use_bias=False), name='td9a'))
    model.add(Bidirectional(LSTM(shape['n_hidden'], return_sequences=True,
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout']), name='bidi9a'))
    # model.add(GlobalMaxPool1D())
    # model.add(BatchNormalization())
    # model.add(Dropout(settings['dropout'] / 2.0))

    # model.add(TimeDistributed(Dense(shape['n_hidden'], use_bias=False), name='td9b'))
    model.add(Bidirectional(LSTM(shape['n_hidden'] // 2, return_sequences=True,
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout']), name='bidi9b'))
    model.add(Bidirectional(LSTM(shape['n_hidden'] // 2, return_sequences=True,
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout']), name='bidi9c'))
    model.add(GlobalMaxPool1D(name='mp9'))
    model.add(BatchNormalization(name='bn9'))
    model.add(Dropout(settings['dropout'] / 2.0, name='drop9b'))

    model.add(Dense(shape['n_class'], activation='sigmoid', name='den9b'))
    xprint('build_lstm11: embeddings=%s shape=%s' % (dim(embeddings), shape))
    return model


def build_lstm12(embeddings, shape, settings):
    """build_lstm6 with more dropout"""
    model = Sequential()
    model.add(
        Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            input_length=shape['max_length'],
            trainable=False,
            weights=[embeddings],
            mask_zero=False
        )
    )
    model.add(TimeDistributed(Dense(shape['n_hidden'], use_bias=False), name='td6'))
    model.add(Bidirectional(LSTM(shape['n_hidden'], return_sequences=True,
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout'])))
    model.add(GlobalMaxPool1D())
    model.add(BatchNormalization())
    model.add(Dropout(settings['dropout']))
    model.add(Dense(shape['n_class'], activation='sigmoid'))
    xprint('build_lstm12: embeddings=%s shape=%s' % (dim(embeddings), shape))
    return model


def build_gpu13(embeddings, shape, settings):
    inp = Input(shape=(shape['max_length'], ))
    # x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            trainable=False,
            weights=[embeddings])(inp)
    x = SpatialDropout1D(settings['dropout'])(x)
    x = Bidirectional(GRU(shape['n_hidden'], return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(6, activation="sigmoid")(conc)

    model = Model(inputs=inp, outputs=outp)

    return model


def build_gpu14(embeddings, shape, settings):
    inp = Input(shape=(shape['max_length'], ))
    # x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            trainable=False,
            weights=[embeddings])(inp)
    x = SpatialDropout1D(settings['dropout'])(x)
    x = Bidirectional(GRU(shape['n_hidden'], return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    conc = BatchNormalization()(conc)
    conc = Dropout(settings['dropout'])(conc)
    outp = Dense(6, activation="sigmoid")(conc)

    model = Model(inputs=inp, outputs=outp)

    return model


build_lstm = {
    1: build_lstm1,
    2: build_lstm2,
    3: build_lstm3,
    4: build_lstm4,
    5: build_lstm5,
    6: build_lstm6,
    7: build_lstm7,
    8: build_lstm8,
    9: build_lstm9,
    10: build_lstm10,
    11: build_lstm11,
    12: build_lstm12,
    13: build_gpu13,
    14: build_gpu14,
}


def compile_lstm(model, learn_rate, frozen):
    if not frozen:
        for layer in model.layers:
            layer.trainable = True

    model.compile(optimizer=Adam(lr=learn_rate), loss='binary_crossentropy', metrics=['accuracy'])
    xprint('compile_lstm: learn_rate=%g' % learn_rate)
    model.summary(print_fn=xprint)
    return model


