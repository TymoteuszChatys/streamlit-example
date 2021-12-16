import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier,SGDClassifier, Perceptron
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer


def predict(question,answer,final_models,model_cvs):
    #We don't use question in our model. 
    #Focus on the answer itself.
    review = re.sub("[^a-zA-z]", ' ', answer)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = " ".join(review)
    review = [review]

    final_results = []
    
    for model,cv in zip(final_models,model_cvs):
        result = cv.transform(review)
        final_results.append((model.predict(result)[0]))
    
    #####If any models say the answer is unhelpful, return unhelpful
    print(final_results)
    print(" ")
    if any(final_results) == 1:
        return "not helpful"
    else:
        return "helpful!"


def start():
    dataset = load_data()
    corpus = populate_corpus(dataset)
    models,features = prepare_models()
    final_models,model_cvs = train_models(models,features,dataset,corpus)
    return final_models,model_cvs


def load_data():
    dataset = pd.read_csv("DATA_EDITED.csv")
    dataset = dataset.dropna()
    dataset = dataset.reset_index(drop=True)

    return dataset

def populate_corpus(dataset):
    corpus = []
    for i in range(len(dataset['answer'])):
        review = re.sub("[^a-zA-z]", ' ', dataset["answer"][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = " ".join(review)
        corpus.append(review)

    return corpus

def prepare_models():
    models = []
    features = []

    models.append(GaussianNB())
    features.append(350)

    models.append(LogisticRegression())
    features.append(325)

    models.append(RidgeClassifier())
    features.append(350)

    models.append(RidgeClassifier())
    features.append(175)

    models.append(SGDClassifier(loss="epsilon_insensitive"))
    features.append(300)

    models.append(SGDClassifier(loss="epsilon_insensitive"))
    features.append(150)

    return models,features

def train_models(models,features,dataset,corpus):
    final_models = []
    model_cvs = []

    for model,n_features in zip(models,features):
        cv = CountVectorizer(max_features = n_features)
        X_pre = cv.fit_transform(corpus)
        X = X_pre.toarray()
        y = dataset.tag

        THE_MODEL = RidgeClassifier()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify=y)
        THE_MODEL.fit(X_train, y_train)
        y_pred = THE_MODEL.predict(X_test)

        """
        print("Accuracy")
        print(accuracy_score(y_train, THE_MODEL.predict(X_train)))
        print(accuracy_score(y_test, THE_MODEL.predict(X_test)))
        print(" ")
        print("F1 Score")
        """
        f1_score_train = f1_score(y_train, THE_MODEL.predict(X_train))
        f1_score_test = f1_score(y_test, THE_MODEL.predict(X_test))
        """
        print(f1_score_train)
        print(f1_score_test)
        """
        final_models.append(THE_MODEL)
        model_cvs.append(cv)

    return final_models,model_cvs
