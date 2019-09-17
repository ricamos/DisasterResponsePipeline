import argparse

import pandas as pd
from datetime import date, datetime
from sqlalchemy import create_engine

# python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

# Configure ArgumentParser 
parser = argparse.ArgumentParser(description = 'Train an AI model.')


parser.add_argument('disaster_messages', action = 'store', help = 'A csv file with disaster menssages.')
parser.add_argument('disaster_categories', action = 'store', help = 'A csv file with disaster menssages.')

arguments = parser.parse_args()

def load_data(messages, categories):

    # load messages dataset
    messages = pd.read_csv(messages)
    categories = pd.read_csv(categories)

    # return merge datasets
    return  messages.merge(categories, on=["id"])


def find_outliers(df):
    outliers_columns = {}
    for x in df.columns[4:]:
        row = df.loc[(df[x] != 0) & (df[x] != 1) ].shape[0]
        if row:
            outliers_columns.update({x : row})
    
    if outliers_columns:
        #print(outliers_columns)
        return outliers_columns
    else:
        print("No outliers")


def drop_outliers(df, outliers_columns):

    outliers_index = []
    for x in outliers_columns:
        index = df.loc[(df[x] != 0) & (df[x] != 1) ].index.values
        for x in index:
            outliers_index.append(x)

    #print(set(outliers_index))
    
    for x in set(outliers_index):
        try:
            df = df.drop([x])
        except:
            pass
    
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


def man(messages = arguments.disaster_messages, categories = arguments.disaster_categories):
	df = load_data(messages,  categories)
	df = etl_pipeline(df)

	# Drop outliers:
	outliers_columns = find_outliers(df)

	if outliers_columns:
		df = drop_outliers(df, outliers_columns)

	filename = save_sql(df)

	print("ETL_output = {}".format(filename))

man()



