import numpy as np
import pandas as pd
import pickle
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


def predict(question,answer,final_models,model_cvs,names):
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
        final_results.append((round(model.predict(result)[0])))
    

    df = pd.DataFrame(list(zip(names, final_results)),
               columns =['Name', 'Prediction'])
    df.index = df.index + 1

    #####If any models say the answer is unhelpful, return unhelpful
    if any(final_results) == 1:
        return df,"not helpful"
    else:
        return df,"helpful!"


def start():
    final_models,model_cvs,names = load_models()
    return final_models,model_cvs,names


def load_models():
    f = open("models.pkl", "rb")
    models = pickle.load(f)
    cvs = pickle.load(f)
    names = pickle.load(f)
    f.close()

    return models,cvs,names