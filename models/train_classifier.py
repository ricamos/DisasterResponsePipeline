import sys

# Load data
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# NLP pipelines
import nltk
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ML Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split

# 5 test your model
from sklearn.metrics import classification_report

# 6 Improve your model
from sklearn.model_selection import GridSearchCV

# 9. Export your model as a pickle file
from datetime import date, datetime
from pytz import timezone
import pickle


def load_data(db):
    """
    INPUT:
        A database file name like: 'disasterResponse.db'
    
    OUTPUT:
        
        Feature and target variables X and Y
        
        X = df.message.values
        Y =  df[df.columns[4:]].values
        
    """
    engine = create_engine(f'sqlite:///{db}')
    df = pd.read_sql('SELECT * FROM disasterResponse', engine)
    X = df.message.values # Feature
    Y =  df[df.columns[4:]].values # Target
    
    return X, Y, df

def tokenize(text):
     # Normalize 
    text = text.lower().strip() # Tudo minusculo 
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    # Tokenize text 
    words = word_tokenize(text) # Separa as sentenças por palavras

    # Testar com e sem stop words
    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")] # Remove palavras padrão

    # lemmatizer root words
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in words:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip() # Root words
        clean_tokens.append(clean_tok)
        
    return clean_tokens

def build_model():
    """
    Returns a model
    """
    
    # Build a machine learning pipeline
    
    lm  = RandomForestClassifier(n_estimators = 10)
    
    pipeline = Pipeline(
    [("vect", CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ("clf", MultiOutputClassifier(lm)),
    ])
    
     # specify parameters for grid search
#     parameters = {
#     'clf__estimator__min_samples_split': [2, 3, 4]
#     }
    
    parameters = {
        # 'vect__ngram_range': ((1, 1), (1, 2)),
        # 'vect__max_df': (0.5, 0.75, 1.0),
        # 'vect__min_df': (0.1, 0.05),
        # 'vect__max_features': (None, 100, 500, 1000, 5000),
        
        
        # 'clf__estimator__n_estimators': [50, 100, 200],
        # 'clf__estimator__max_depth': [4, 8, 16],
         'clf__estimator__min_samples_leaf': [2, 4, 8],
         'clf__estimator__min_samples_split': [2, 4, 8],
    
    }
    
    # create grid search object
    #par = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=5)
    
    return cv

def save_model(model, model_filepath):
    """
    Save your model as a pikle file.
    
    """
 

    # save the model to disk
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        
        database_filepath, model_filepath = sys.argv[1:]
        
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, df = load_data(database_filepath)
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.33, random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        # print('Evaluating model...')
        # evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('\nPlease provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db\n')

    
# Load
if __name__ == '__main__':
    main()