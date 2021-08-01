'''
       ___       ___           ___     
      /\__\     /\  \         /\__\    
     /:/  /    /::\  \       /::|  |   
    /:/  /    /:/\ \  \     /:|:|  |   
   /:/  /    _\:\~\ \  \   /:/|:|__|__ 
  /:/__/    /\ \:\ \ \__\ /:/ |::::\__\ 
  \:\  \    \:\ \:\ \/__/ \/__/~~/:/  /
   \:\  \    \:\ \:\__\         /:/  / 
    \:\  \    \:\/:/  /        /:/  /  
     \:\__\    \::/  /        /:/  /   
      \/__/     \/__/         \/__/    
'''
# Latin Scansion Model
# Philippe Bors and Luuk Nolden
# Leiden University 2021

# Library Imports
import configparser
import pandas as pd
import numpy as np

# Class Imports
from word2vec.word2vec import Word_vector_creator 
from preprocessor.preprocessor import Text_preprocessor 
import utilities as util
from pedecerto.parser import Pedecerto_parser

import neural_network.parser as nn

########
# MAIN #
########

# Parameters to run each step
run_preprocessor = False
run_pedecerto = True
run_model_generator = False
add_embeddings_to_df = False
run_neural_network = False

use_file = False # Set to true if you want to use the X and y files present in the pickle dir

# Read the config file for later use
cf = configparser.ConfigParser()
cf.read("config.ini")

''' Run the preprocessor on the given text if needed.
This reads the text, cleans it and returns a list of syllables for now
To achieve this, the pedecerto tool is used
'''
if run_preprocessor:
    preprocessor = Text_preprocessor(cf.get('Text', 'name'))
    util.Pickle_write(cf.get('Pickle', 'path'), cf.get('Pickle', 'char_list'), preprocessor.character_list)

# Load the preprocessed text
character_list = util.Pickle_read(cf.get('Pickle', 'path'), cf.get('Pickle', 'char_list'))

''' Now create a dataframe. Containing: syllable, length, vector.
'''
if run_pedecerto:

    parse = Pedecerto_parser(cf.get('Pedecerto', 'path_texts'), -1)  
    util.Pickle_write(cf.get('Pickle', 'path'), cf.get('Pickle', 'pedecerto_df'), parse.df)
    # parse.df.to_csv(cf.get('Pickle', 'pedecerto_df'), index = False, header=True) #FIXME: save to pickle

pedecerto_df = util.Pickle_read(cf.get('Pickle', 'path'), cf.get('Pickle', 'pedecerto_df'))
print(pedecerto_df)

# exit(0)

# Run the model generator on the given list if needed
if run_model_generator:
    # Create a word2vec model from the provided character list
    word2vec_creator = Word_vector_creator(character_list, cf.getint('Word2Vec', 'vector_size'), cf.getint('Word2Vec', 'window_size'))
    util.Pickle_write(cf.get('Pickle', 'path'), cf.get('Pickle', 'word2vec_model'), word2vec_creator.model)

# Load the saved/created model
word2vec_model = util.Pickle_read(cf.get('Pickle', 'path'), cf.get('Pickle', 'word2vec_model'))

# Add the embeddings created by word2vec to the dataframe
if add_embeddings_to_df:

    df = pedecerto_df
    
    # Add syllable vectors to the dataframe using the word2vec model
    df['vector'] = df['syllable']
    for i in range(len(df)):
        try:
            syllable = df["syllable"][i]
            df["vector"][i] = word2vec_model.wv[syllable]
        except:
            IndexError('Syllable has no embedding yet.')

    # df.to_csv(cf.get('Pickle', 'embedding_df'), index = False, header=True) #FIXME: save to pickle
    util.Pickle_write(cf.get('Pickle', 'path'), cf.get('Pickle', 'embedding_df'), df)

# Provide the neural network with the dataframe
if run_neural_network:
    
    df = util.Pickle_read(cf.get('Pickle', 'path'), cf.get('Pickle', 'embedding_df'))
    
    X, y = nn.load_data(df, use_file=False) # Either by finalizing parsing or by files


    print("Training data: shape={}".format(X.shape))
    print("Training target data: shape={}".format(y.shape))

    exit(0)

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from matplotlib import pyplot

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    HIDDEN_SIZE = 256

    def create_model(X_train, y_train):
        """Create a neural network with two hidden layers,
             Dependent on the sizes of the training data

        Args:
            X_train (array): .
            y_train (array): .

        Returns:
            ...
        """    
        model = Sequential()
        model.add(Dense(HIDDEN_SIZE, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dense(HIDDEN_SIZE, activation='relu'))
        model.add(Dense(y_train.shape[1], activation='sigmoid'))
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        return model

    # Split test and train set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Scale data
    #scaler = MinMaxScaler()
    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)

    # Create the model
    model = create_model(X_train, y_train)

    # # Train
    history = model.fit(X_train, y_train, epochs=80, batch_size=1, validation_data=(X_test, y_test), shuffle=True)


    _, train_accuracy = model.evaluate(X_train, y_train)
    _, test_accuracy = model.evaluate(X_test, y_test)

    print('Accuracy (training): %.2f' % (train_accuracy * 100))
    print('Accuracy (testing): %.2f' % (test_accuracy * 100))


