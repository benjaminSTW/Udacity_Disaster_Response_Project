import sys
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import  classification_report
import pickle


def load_data(database_filepath):
    '''
    Load the data message and categories

            Parameters:
                    messages_filepath (string): filepath of the messages CSV
                    categories_filepath (string): filepath of the categories CSV

            Returns:
                    X: 
                    Y:
                    categories : 
    '''
    
    # load data from database
    database_filepath= 'sqlite:///'+database_filepath
    engine = create_engine(database_filepath)
    df = pd.read_sql_table('TF_Messages',con=engine)
    #creation of our X variable containing only messages
    X = df['message']
    #creation of our X variable with all the columns except the messages
    Y =df.drop(columns=['message'])
    categories = Y.columns
    return X,Y,categories

def tokenize(text):
    '''
    Returns the list of cleaned words from the input message

            Parameters:
                    text (string): the message (text) to clean

            Returns:
                    clean_tokens (array): array of the cleaned words from the input message.
    '''
    tokens = word_tokenize(text) #We split the message into word
    lemmatizer = WordNetLemmatizer() #Instantiation of WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip() # Lemmatize, lower_cap and remove spaces
        clean_tokens.append(clean_tok) #append clean words to the results clean_tokens

    return clean_tokens


def build_model():
    '''
    Create the model using a pipeline

            Parameters:
                   None

            Returns:
                   pipeline_GB : model create using a pipeline
    '''

    pipeline_GB = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('ada_clf', MultiOutputClassifier(AdaBoostClassifier(learning_rate=0.5,n_estimators=150)))])
    return pipeline_GB

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate the model

            Parameters:
                   model : model to be evaluated
                   X_test : X dataset for evaluation
                   Y_test : Y dataset for evaluation
                   category_names : list of the categories

            Returns:
                   None
    '''
    y_pred = model.predict(X_test)
    for index,c in enumerate(category_names):
        print ("---" +c+ "---")
        print(classification_report (Y_test.iloc[:,index].values, y_pred[:,index]))


def save_model(model, model_filepath):
    '''
    Saving the model

            Parameters:
                   model : model to be saved
                   model_filepath : path where to save the model

            Returns:
                   None
    '''
    with open (model_filepath,'wb') as f:
        pickle.dump(model,f)


def main():
    '''main function to build, train, evaluate and save the model
    
         Parameters :
                   database_filepath : database path where input message are stored
                   model_filepath : path where the model shoulbe be saved.

            Returns:
                    model is saved under the filename defined by input parameters'''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()