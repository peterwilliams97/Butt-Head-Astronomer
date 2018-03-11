#!/bin/bash

pushd ~/data

wget http://nlp.stanford.edu/data/glove.twitter.27B.zip
wget http://nlp.stanford.edu/data/glove.6B.zip
wget http://nlp.stanford.edu/data/glove.840B.300d.zip

unzip glove.twitter.27B.zip
unzip glove.6B.zip
unzip glove.840B.300d.zip

popd
echo done!
