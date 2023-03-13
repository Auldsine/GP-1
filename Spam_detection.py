import pandas as pd
import re
import string
import numpy as np
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS

data = pd.read_csv("spam.csv",encoding = "'latin'")

data.tail()

data["text"] = data.v2
data["spam"] = data.v1

from sklearn.model_selection import train_test_split
emails_train, emails_test, target_train, target_test = train_test_split(data.text,data.spam,test_size = 0.2)

emails_test.tail()

data.info

emails_train.shape

# Preprocessing

def remove_hyperlink(word):
    return  re.sub(r"http\S+", "", word)

def to_lower(word):
    result = word.lower()
    return result

def remove_number(word):
    result = re.sub(r'\d+', '', word)
    return result

def remove_punctuation(word):
    result = word.translate(str.maketrans(dict.fromkeys(string.punctuation)))
    return result

def remove_whitespace(word):
    result = word.strip()
    return result

def replace_newline(word):
    return word.replace('\n','')



def clean_up_pipeline(sentence):
    cleaning_utils = [remove_hyperlink,
                      replace_newline,
                      to_lower,
                      remove_number,
                      remove_punctuation,remove_whitespace]
    for o in cleaning_utils:
        sentence = o(sentence)
    return sentence

x_train = [clean_up_pipeline(o) for o in emails_train]
x_test = [clean_up_pipeline(o) for o in emails_test]

x_train[0]


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_y = le.fit_transform(target_train.values)
test_y = le.transform(target_test.values)

train_y

# Tokenize

## some config values 
embed_size = 100 # how big is each word vector
max_feature = 50000 # how many unique words to use (i.e num rows in embedding vector)
max_len = 2000 # max number of words in a question to use


from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=max_feature)

tokenizer.fit_on_texts(x_train)

x_train_features = np.array(tokenizer.texts_to_sequences(x_train))
x_test_features = np.array(tokenizer.texts_to_sequences(x_test))

x_train_features[0]


# Padding

from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train_features = pad_sequences(x_train_features,maxlen=max_len)
x_test_features = pad_sequences(x_test_features,maxlen=max_len)
x_train_features[0]


# Model

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional
from keras.models import Model

# create the model
import tensorflow as tf
from keras.layers import Dense,LSTM, Embedding, Dropout, Activation, Bidirectional

#size of the output vector from each layer
embedding_vecor_length = 32

#Creating a sequential model
model = tf.keras.Sequential()

#Creating an embedding layer to vectorize
model.add(Embedding(max_feature, embedding_vecor_length, input_length=max_len))

#Addding Bi-directional LSTM
model.add(Bidirectional(tf.keras.layers.LSTM(64)))

#Relu allows converging quickly and allows backpropagation
model.add(Dense(16, activation='relu'))

#Deep Learninng models can be overfit easily, to avoid this, we add randomization using drop out
model.add(Dropout(0.1))

#Adding sigmoid activation function to normalize the output
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


history = model.fit(x_train_features, train_y, batch_size=2048, epochs=1, validation_data=(x_test_features, test_y))

from  matplotlib import pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid()
plt.show()

from sklearn.metrics import confusion_matrix,f1_score, precision_score,recall_score

y_predict  = [1 if o>0.5 else 0 for o in model.predict(x_test_features)]

cf_matrix =confusion_matrix(test_y,y_predict)

import seaborn as sns
import matplotlib.pyplot as plt     

ax= plt.subplot()
sns.heatmap(cf_matrix, annot=True, ax = ax,cmap='Blues',fmt=''); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Not Spam', 'Spam']); ax.yaxis.set_ticklabels(['Not Spam', 'Spam']);

tn, fp, fn, tp = confusion_matrix(test_y,y_predict).ravel()

# ___________________________________________________________#

def preprocess_test_data(sentence):
    sentence_cleaned = clean_up_pipeline(sentence)
    sentence_sequence = np.array(tokenizer.texts_to_sequences([sentence_cleaned]))
    sentence_padded = pad_sequences(sentence_sequence, maxlen=max_len)
    return sentence_padded

def paragraph_prediction(sentence):
    preprocessed_sentence = preprocess_test_data(sentence)
    prediction = model.predict(preprocessed_sentence)
    if prediction>0.5:
        return "spam"
    else:
        return "not spam"
    
paragraph_prediction("Ok lar... Joking wif u oni...")

# ___________________________________________________________#

print("Precision: {:.2f}%".format(100 * precision_score(test_y, y_predict)))
print("Recall: {:.2f}%".format(100 * recall_score(test_y, y_predict)))
print("F1 Score: {:.2f}%".format(100 * f1_score(test_y,y_predict)))

f1_score(test_y,y_predict)

