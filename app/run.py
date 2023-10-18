import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar 
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')

df = pd.read_sql_table('TF_Messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    type_counts = df.drop(columns=['message']).sum()
    type_names = list(type_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=type_names,
                    y=type_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message types',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Type of message"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[1:], classification_labels))
    print (classification_labels)
    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()