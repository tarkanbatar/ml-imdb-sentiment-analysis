import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import tensorflow as tf

from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import SimpleRNN, Dense, Activation

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import model_selection
from sklearn.svm import SVC

(X_train, Y_train), (X_test, Y_test) = imdb.load_data(path="ibdb.npz",
                                                      num_words=None,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)

print("X train shape: ", X_train.shape)
print("Y train shape: ", Y_train.shape)
print()

print("X test shape: ", X_test.shape)
print("Y test shape: ", Y_test.shape)
print()

# EDA

print("X train values: ", X_train[0:10], "\n")
print("Y train values: ", Y_train[0:10], "\n")

print("Y train values: ", np.unique(Y_train))
print("Y test values: ", np.unique(Y_test), "\n")

unique, counts = np.unique(Y_train, return_counts=True)
print("Y train distribution: ", dict(zip(unique, counts)))

unique, counts = np.unique(Y_test, return_counts=True)
print("Y test distribution: ", dict(zip(unique, counts)), "\n")

plt.figure()
sns.countplot(Y_train)
plt.xlabel("Classes")
plt.ylabel("Freq")
plt.title("Y train")

plt.figure()
sns.countplot(Y_test)
plt.xlabel("Classes")
plt.ylabel("Freq")
plt.title("Y test")

review_len_train = []
review_len_test = []

for i, ii in zip(X_train, X_test):
    review_len_train.append(len(i))
    review_len_test.append(len(ii))

sns.distplot(review_len_train, hist_kws={"alpha": 0.3})
sns.distplot(review_len_test, hist_kws={"alpha": 0.3})

print("Train mean:", np.mean(review_len_train))
print("Train median:", np.median(review_len_train))

print("Test mean:", np.mean(review_len_test))
print("Test median:", np.median(review_len_test), "\n")


# Number of words
word_index = imdb.get_word_index()
print(type(word_index))
print(len(word_index))

for keys, values in word_index.items():
    if values == 22:
        print(keys)

def commentPreview(index=24):
    reverse_index = dict([(value, key) for (key, value) in word_index.items()])
    decode_review = " ".join([reverse_index.get(i - 3, "!") for i in X_train[index]])
    print(decode_review)
    print("This comment is: ", Y_train[index])
    return decode_review


commentPreview(518)
print()

# Preprocess

num_words = 15000
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=num_words)

X_train = pad_sequences(X_train, maxlen=130)
X_test = pad_sequences(X_test, maxlen=130)

k = 0
print()
for i in X_train[0:10]:
    print("Length of ", (k + 1), ". varible is: ", len(i))
    k += 1

print()
decoded_review = commentPreview(20)

# RNN
maxlen = 130
rnn = Sequential()
rnn.add(Embedding(num_words, 32, input_length=len(X_train[0])))
rnn.add(SimpleRNN(16, input_shape=(num_words, maxlen), return_sequences=False, activation="relu"))
rnn.add(Dense(1))
rnn.add(Activation("sigmoid"))

print(rnn.summary())
rnn.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

history = rnn.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=5, batch_size=128, verbose=1)

score = rnn.evaluate(X_test, Y_test)
print("Accuracy: %", score[1] * 100)

# Naive Bayes

nbAccuracy = 0
for i in range(10):
    kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=i)
    gnb = GaussianNB()

    results = model_selection.cross_validate(gnb, X_train, Y_train, cv=kfold, scoring=['accuracy'])
    nbAccuracy = nbAccuracy + results['test_accuracy'].mean()

print("Naive Bayes Accuracy: %", (nbAccuracy * 10))

plt.figure()
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Test")
plt.title("Acc")
plt.ylabel("Acc")
plt.xlabel("Epochs")
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Test")
plt.title("Loss")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()


