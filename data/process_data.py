# Imports
import pandas as pd
from datetime import date, datetime
from sqlalchemy import create_engine
import sys

# DEFs
def load_data(messages, categories):

    # load messages dataset
    messages = pd.read_csv(messages)
    categories = pd.read_csv(categories)

    # return merge datasets
    return  messages.merge(categories, on=["id"])

def only_ony_class(df):
    """
    Find and drop columns with number of labels other than 2.
    """
    col_one_class = []
    for col in df.columns[4:]:
        if df[col].unique().shape[0] != 2:
            col_one_class.append(col)
            df = df.drop([col], axis=1)
    print(col_one_class)
    return df

def drop_outliers(df):
    """
    Input:
        The Data Frame and a dictionary with column names that contain different values.
        
    Output:
        A new dataframe without rows with diffetents values
    """
    
    outliers_index = []
    for col in df.columns[4:]:
        index = df.loc[(df[col] != 0) & (df[col] != 1)].index.values
        for x in index:
             outliers_index.append(x)

    for row in df.index:
        if df[df.columns[4:]].loc[row].unique().shape[0]<2:
            outliers_index.append(row)

    #print(set(outliers_index))
    
    for x in set(outliers_index):
            df = df.drop([x])
            
    return df

def etl_pipeline(df):
 	
 	# create a dataframe of the 36 individual category columns
	categories = df["categories"].str.split(";", expand=True)

	# select the first row of the categories dataframe
	row = categories.iloc[0]
	category_colnames = row.replace('-.+', '', regex=True)

	# rename the columns of `categories`
	categories.columns = category_colnames

	# 4 Convert category values to just numbers 0 or 1
	for column in categories:
		# set each value to be the last character of the string
		categories[column] = categories[column].str.get(-1)

		# convert column from string to numeric
		categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from `df`
	df = df.drop("categories", axis=1)

	# concatenate the original dataframe with the new `categories` dataframe
	df = pd.concat([df, categories], axis=1)
    
     # drop duplicates
	df = df.drop_duplicates()
    
	return df

def save_sql(df):
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    filename = f'disasterResponse_{dt_string}.db'

    engine = create_engine(f'sqlite:///{filename}')
    df.to_sql('disasterResponse', engine, index=False)

    return filename

# App
def main():
	if len(sys.argv) == 3:

		messages, categories = sys.argv[1:]

		print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(messages, categories))

		df = load_data(messages, categories)
		

		print('Cleaning data...')
		df = etl_pipeline(df)

		df = only_ony_class(df) # Deve ser executada antes de drop_outliers

		df = drop_outliers(df)

		print(df.shape)

		filename = save_sql(df)
		print('Saving data...\n    DATABASE: {}'.format(filename))

	else:
		print('\nPlease provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, '\
              'The  cleaned data will save in a database file.\
              \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv \n')

# RUN
if __name__ == '__main__':
    main()
