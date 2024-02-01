# Case folding
import re
def case_folding(question_text):
    question_text = re.sub(r'http\S+', ' ', question_text)
    question_text = re.sub(r'\d+', '', question_text)
    question_text = str.lower(question_text)
    return question_text

# Tokenizing
from nltk.tokenize import RegexpTokenizer
def tokenizing(question_text):
    tokenizer = RegexpTokenizer(r'\w+')
    question_text = tokenizer.tokenize(question_text)
    return question_text

# Stemmming
from nltk.stem import PorterStemmer
def stemming(question_text):
    stemmer = PorterStemmer()
    question_text = [stemmer.stem(word) for word in question_text]
    return question_text

from nltk.stem import WordNetLemmatizer
def lemmatizing(question_text):
    lemmatizer = WordNetLemmatizer()
    question_text = [lemmatizer.lemmatize(word) for word in question_text]
    return question_text

# Stop words
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
def stopword_remove(question_text):
    stop_words = stopwords.words('english')
    question_text = [word for word in question_text if word not in stop_words]
    return question_text

# Combine all preprocessing
def preprocessing_text_with_stemming(question_text):
    question_text = case_folding(question_text)
    question_text = tokenizing(question_text)
    question_text = stemming(question_text)
    question_text = stopword_remove(question_text)
    return question_text

def preprocessing_text_with_lemma(question_text):
    question_text = case_folding(question_text)
    question_text = tokenizing(question_text)
    question_text = lemmatizing(question_text)
    question_text = stopword_remove(question_text)
    return question_text

def preprocessing_text_no_stopword(question_text):
    question_text = case_folding(question_text)
    question_text = tokenizing(question_text)
    question_text = stemming(question_text)
    return question_text

def preprocessing_text_no_stem(question_text):
    question_text = case_folding(question_text)
    question_text = tokenizing(question_text)
    question_text = stopword_remove(question_text)
    return question_text