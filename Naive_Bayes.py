import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
import pandas as pd
import string
from nltk.stem import PorterStemmer
import re
import seaborn as sns
from textblob import TextBlob
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report




class Sentiment_Analysis:
    def NaiveBayes(df):
        def getSubjectivity(review):
            return TextBlob(review).sentiment.subjectivity
            # function to calculate polarity

        def getPolarity(review):
            return TextBlob(review).sentiment.polarity

        # function to analyze the reviews
        def analysis(score):
            if score < 0:
                return 'Negative'
            elif score == 0:
                return 'Neutral'
            else:
                return 'Positive'

        df['Subjectivity'] = df['Stem_String'].apply(getSubjectivity)
        df['Polarity'] = df['Stem_String'].apply(getPolarity)
        df['Analysis'] = df['Polarity'].apply(analysis)
        plt.figure(figsize=(15, 7))
        plt.subplot(1, 3, 1)
        plt.title("Naive Bayes results")
        tb_counts = df.Analysis.value_counts()
        plt.pie(tb_counts.values, labels=tb_counts.index, explode=(0, 0, 0.25), autopct='%1.1f%%', shadow=False)
        plt.show()

        number = LabelEncoder()

        df['Polarity'] = number.fit_transform(df['Polarity'])

        # Polarity
        features = ["Polarity"]

        target = "Polarity"
        features_train, features_test, target_train, target_test = train_test_split(df[features], df[target],
                                                                                    test_size=.2000,
                                                                                    random_state=35)

    dfa = pd.read_csv('VaccTest13Sentiment.csv', delimiter=',', encoding='UTF-8')
    NaiveBayes(dfa)


cr = Sentiment_Analysis()
df = cr.dfa.head()

df = pd.read_csv('VaccTest13Sentiment.csv', delimiter=',', encoding='UTF-8')
sns.pairplot(df)
labelencoder = LabelEncoder()

df['Analysis'] = labelencoder.fit_transform(df['Analysis'])

df['Stem_Tweets'] = labelencoder.fit_transform(df['Stem_Tweets'])
df['Stem_String'] = labelencoder.fit_transform(df['Stem_String'])
df['keywords'] = labelencoder.fit_transform(df['keywords'])

X = df[['Unnamed: 0', 'Stem_Tweets', 'Stem_String', 'keywords', 'Subjectivity', 'Polarity']]
y = df['Analysis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70, random_state=60)

GaussianNB = GaussianNB()
GaussianNB.fit(X_train, y_train)
prediction = GaussianNB.predict(X_test)

print("Accuracy", accuracy_score(y_test, prediction) * 100)
print(classification_report(y_test, prediction))

classification_report = pd.DataFrame(classification_report(y_test, prediction, output_dict=True)).transpose()
classification_report.to_csv('result.csv')

