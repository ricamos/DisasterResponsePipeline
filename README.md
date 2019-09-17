# Write A Data Science Blog Post 

## Table of Contents:

1. [Motivation](#motivation)
2. [File description](#file)
3. [How to interact with your project](#interact)
4. [Licensing](#licensing)
5. [Authors](#author)
6. [Acknowledgements](#ack)

## Motivation <a name="motivation"></a>

This project is part of Udacity Data Scientist Nanodegree.

In this project, I'll apply data engineering skills to analyze disaster data from [Figure Eight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages.

In the Project Workspace, I'll find a data set containing real messages that were sent during disaster events. I will be creating a machine learning pipeline to categorize these events so that i can send the messages to an appropriate disaster relief agency.

The project include a web app where an emergency worker can input a nem message and get classification results in several categories.

## File description <a name="file"></a>

ETL Pipeline Preparation.ipynb - A jupyter file with an ETL Pipeline Sketch

process_data.py - A data cleaning pipeline.  Loads the messages and categories datasets. Merges the two datasets. Cleans the data and Stores it in a SQLite database.

train_classifier.py - A machine learning pipeline. Loads data from the SQLite database. Splits the dataset into training and test sets. Builds a text processing and machine learning pipeline. Trains and tunes a model using GridSearchCV. Outputs results on the test set. Exports the final model as a pickle file.

## How to interact with your project <a name="interact"></a>

## Licensing <a name="licensing"></a>
[License file](https://github.com/ricamos/DisasterResponsePipeline/blob/master/LICENSE)

## Authors <a name="author"></a>
Ricardo Coelho

## Acknowledgements <a name="ack"></a>
The project will also be graded based on the following:

- Use of Git and Github
- Strong documentation
- Clean and modular code