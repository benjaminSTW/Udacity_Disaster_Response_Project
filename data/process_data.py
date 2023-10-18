
import sys
import pandas as pd   
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load the data message and categories

            Parameters:
                    messages_filepath (string): filepath of the messages CSV
                    categories_filepath (string): filepath of the categories CSV

            Returns:
                    df (dataframe): dataframe containing messages and categories
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    #load categories dataset
    categories = pd.read_csv(categories_filepath)
    #we merge the two dataset based on the column id
    df = messages.merge(categories,on='id')
    return df

def clean_data(df):
    '''
    clean the data stored in the input dataframe

            Parameters:
                    df (dataframe): dataframe to be cleaned

            Returns:
                    df_cleaned (dataframe): cleaned dataframe
    '''
    categories = df.categories.str.split(';',expand=True)
    # select the first row of the categories dataframe
    row =categories.iloc[0]
    #Based on row values we adjust the columns names
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    #Convert category values to 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).apply(lambda x:x[-1])
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    #related columns contains data with the value equals to 2, we need to fix it
    categories.loc[categories['related']==2,"related"]=1
    #We drop the former categories column
    df.drop(columns=['categories'],inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    #we drop duplicates
    df.drop_duplicates(inplace=True)
    #Remove unnecessary columns
    df_cleaned =df.drop(columns=['original','genre','id'])
    return df_cleaned

def save_data(df, database_filename):
    '''
    save the dataframe in the database in the table TF_Messages

            Parameters:
                    df (dataframe): dataframe to be saved
                    database_filename : filenam of the database

            Returns:
                    None
    '''
    database_filename = "sqlite:///" + database_filename
    engine = create_engine(database_filename)
    df.to_sql('TF_Messages', engine, index=False,if_exists='replace')  


def main():
    '''
    Main function to run the etl process.

            Parameters:
                    messages_filepath: filepath of raw messages
                    categories_filepath : filenam of raw categories
                    database_filepath : filenam of the database to store cleaned messages
            Returns:
                    database saved with cleaned messages
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()