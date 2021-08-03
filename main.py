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
from progress.bar import Bar

# Class Imports
from word2vec import Word_vector_creator 
from preprocessor import Text_preprocessor 
from pedecerto.textparser import Pedecerto_parser #FIXME: needs a nice rename
from neuralnetwork import Neural_network_handler

import utilities as util

########
# MAIN #
########

# Parameters to run each step
run_preprocessor = False
run_pedecerto = False
run_model_generator = False
add_embeddings_to_df = False
run_neural_network = True

# Read the config file for later use
cf = configparser.ConfigParser()
cf.read("config.ini")

print('\n')

''' Run the preprocessor on the given text if needed.
This reads the text, cleans it and returns a list of syllables for now
To achieve this, the pedecerto tool is used
'''
if run_preprocessor:
    print('Running preprocessor')
    preprocessor = Text_preprocessor(cf.get('Text', 'name'))
    util.Pickle_write(cf.get('Pickle', 'path'), cf.get('Pickle', 'char_list'), preprocessor.character_list)

# Load the preprocessed text
character_list = util.Pickle_read(cf.get('Pickle', 'path'), cf.get('Pickle', 'char_list'))

''' Now create a dataframe. Containing: syllable, length, vector.
'''
if run_pedecerto:
    print('Running pedecerto parser')
    parse = Pedecerto_parser(cf.get('Pedecerto', 'path_texts'), -1)  
    util.Pickle_write(cf.get('Pickle', 'path'), cf.get('Pickle', 'pedecerto_df'), parse.df)

pedecerto_df = util.Pickle_read(cf.get('Pickle', 'path'), cf.get('Pickle', 'pedecerto_df'))

# Run the model generator on the given list if needed
if run_model_generator:
    print('Running Word2Vec model generator')
    # Create a word2vec model from the provided character list
    word2vec_creator = Word_vector_creator(character_list, cf.getint('Word2Vec', 'vector_size'), cf.getint('Word2Vec', 'window_size'))
    util.Pickle_write(cf.get('Pickle', 'path'), cf.get('Pickle', 'word2vec_model'), word2vec_creator.model)

# Load the saved/created model
word2vec_model = util.Pickle_read(cf.get('Pickle', 'path'), cf.get('Pickle', 'word2vec_model'))

# Add the embeddings created by word2vec to the dataframe
if add_embeddings_to_df:
    print('Adding embeddings to the dataframe')
    df = pedecerto_df
    
    # Add syllable vectors to the dataframe using the word2vec model
    df['vector'] = df['syllable']
    # for i in range(len(df)):
    for i in Bar('Processing').iter(range(len(df))): # This is a very nice progress bar I like very much
        try:
            syllable = df["syllable"][i]
            df["vector"][i] = word2vec_model.wv[syllable]
        except:
            IndexError('Syllable has no embedding yet.')

    # df.to_csv(cf.get('Pickle', 'embedding_df'), index = False, header=True) #FIXME: save to pickle
    util.Pickle_write(cf.get('Pickle', 'path'), cf.get('Pickle', 'embedding_df'), df)

# Provide the neural network with the dataframe
if run_neural_network:
    print('Running the neural network generation')

    df = util.Pickle_read(cf.get('Pickle', 'path'), cf.get('Pickle', 'embedding_df'))

    nn = Neural_network_handler(df)  





