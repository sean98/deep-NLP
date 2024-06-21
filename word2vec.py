import functools
import operator
import requests
import numpy as np


def random_vocabulary(sentences):
    words = {w[0].upper() for w in functools.reduce(operator.concat, sentences)}
    return {w: np.random.random(50) for w in words}


def create_dataset(sentences, vocabulary, empty_embedding=np.zeros(50), default_embedding=np.ones(50), windows_size=2):
    dataset = []
    for sentence in sentences:
        for i in range(len(sentence)):
            vec = []
            for j in range(-windows_size, windows_size + 1):
                if (0 > i + j) or (i + j >= len(sentence)):
                    vec.append(empty_embedding)
                else:
                    vec.append(vocabulary.get(sentence[i + j][0].upper(), default_embedding))
            dataset.append((np.asfarray(vec).flatten(), sentence[i][1]))
    return dataset


def download_word_vector():
    _res = requests.get('https://u.cs.biu.ac.il/~89-687/ass2/vocab.txt')
    _words = _res.text.strip().split('\n')

    _res = requests.get('https://u.cs.biu.ac.il/~89-687/ass2/wordVectors.txt')
    _vectors = [np.asfarray(v.strip().split(' ')) for v in _res.text.strip().split('\n')]

    return dict(zip(_words, _vectors))
