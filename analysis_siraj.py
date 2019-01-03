import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb


train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000,
                               valid_portion=0.1)

print(train.type)

trainX, trainY = train
testX, testY = test

