import numpy as np
import pandas as pd
import pickle
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
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


def predict(question,answer,final_models,model_cvs,names,tokenizer,vocab_list,cnn_model):
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
    
    cnn_result = run_cnn(answer,vocab_list,tokenizer,cnn_model)[0][0]
    names.append("Robinho Neural Net")
    final_results.append(float(cnn_result))

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
    f = open("Models/models.pkl", "rb")
    models = pickle.load(f)
    cvs = pickle.load(f)
    names = pickle.load(f)
    f.close()

    return models,cvs,names


def load_nn():
    tokenizer = pickle.load(open('Models/NN/tokenizer.pickle', 'rb'))    
    vocab_list = pickle.load(open('Models/NN/cnn_vocab_list.sav', 'rb'))
    cnn_model = pickle.load(open('Models/NN/CNN_model.sav', 'rb'))


    return tokenizer,vocab_list,cnn_model

def clean_ans_cnn(ans, vocab):
	#Splits answer into list of words
    words = ans.split()
 
	#Removes punctuation from the list of words (First line makes a map from list of all punctuation to '')
    rem_punc = str.maketrans('', '', string.punctuation)
    words = [w.translate(rem_punc) for w in words]

	#Filter out words in the answer that don't appear in our vocab list.
    words = [w for w in words if w in vocab]
    words = ' '.join(words)

    return words


def run_cnn(ans,vocab_list, tokenizer,cnn_model):
    ans = [clean_ans_cnn(str(ans),vocab_list)]
    #Sequence encode the words in the training set - I.e give each unique word/token a numeric value
    encoded_docs = tokenizer.texts_to_sequences(ans)
    #Pad sequences - for every answer- the encoding must be the same length so we add 0's to the end of the answers so that they are all equal in length.
    ans = pad_sequences(encoded_docs, maxlen=393, padding='post')
    cnn_pred = (cnn_model.predict(ans) > 0.5).astype('int32')

    return cnn_pred