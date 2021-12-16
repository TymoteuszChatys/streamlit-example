import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gzip
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier,SGDClassifier, Perceptron
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt



dataset = pd.read_csv("DATA_EDITED.csv")
dataset = dataset.dropna()
dataset = dataset.reset_index(drop=True)