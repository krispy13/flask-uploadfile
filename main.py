import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import wordnet
import string, re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib
import pickle

df = pd.read_csv('spam.csv', encoding="latin-1")
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
df['v1'] = df['v1'].map({'ham': 0, 'spam': 1})
df['v2'] = df['v2'].apply(lambda x: x.lower())


def remove_punctuation(text):
    punctuationfree = "".join([i for i in text if i not in string.punctuation])
    return punctuationfree


df['v2'] = df['v2'].apply(lambda x: remove_punctuation(x))
stopwords = nltk.corpus.stopwords.words('english')
df['v2'] = df['v2'].apply(lambda x: re.split(r"\s+", x))


def remove_stopwords(text):
    output = [i for i in text if i not in stopwords]
    return output


df['v2'] = df['v2'].apply(lambda x: remove_stopwords(x))
lemm = nltk.stem.WordNetLemmatizer()

df['v2'] = df['v2'].apply(lambda x: [lemm.lemmatize(word) for word in x])
df['v2'] = df['v2'].apply(lambda x: ' '.join([item for item in x]))

X = df['v2']
y = df['v1']
print(f'The dataset is as follows: \n{df[["v1", "v2"]]}\n')
cv = CountVectorizer()
X = cv.fit_transform(X)
print(f'Original Vector Shape: {X.shape}')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1333)
clf = MultinomialNB()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
y_pred = clf.predict(X_test)
print(f'Classification Report For NB: \n {classification_report(y_test, y_pred)}')

df2 = pd.read_csv('data.csv')
data = df2['v2']
vect = cv.transform(data).toarray()
my_prediction = clf.predict(vect)
print(my_prediction)

pickle.dump(clf, open('model.pkl', 'wb'))
pickle.dump(cv, open('vect.pkl', 'wb'))
