#!/bin/bash

pushd ~/data

mkdir glove.twitter.27
pushd glove.twitter.27
wget http://nlp.stanford.edu/data/glove.twitter.27B.zip
unzip glove.twitter.27B.zip
popd

mkdir glove.6B
pushd glove.6B
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
popd

mkdir glove.840B.300d
pushd glove.840B.300d
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
popd

popd
echo done!
