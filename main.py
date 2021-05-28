#       ___       ___           ___     
#      /\__\     /\  \         /\__\    
#     /:/  /    /::\  \       /::|  |   
#    /:/  /    /:/\ \  \     /:|:|  |   
#   /:/  /    _\:\~\ \  \   /:/|:|__|__ 
#  /:/__/    /\ \:\ \ \__\ /:/ |::::\__\
#  \:\  \    \:\ \:\ \/__/ \/__/~~/:/  /
#   \:\  \    \:\ \:\__\         /:/  / 
#    \:\  \    \:\/:/  /        /:/  /  
#     \:\__\    \::/  /        /:/  /   
#      \/__/     \/__/         \/__/    

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


########
# MAIN #
########

# Temp params
run_preprocessor = False
run_model_generator = False
run_pedecerto = False

# Read the config file for later use
cf = configparser.ConfigParser()
cf.read("config.ini")

# Run the preprocessor on the given text if needed
if run_preprocessor:
    preprocessor = Text_preprocessor(cf.get('Text', 'name'))
    util.Pickle_write(cf.get('Pickle', 'path'), cf.get('Pickle', 'char_list'), preprocessor.character_list)

# Load the preprocessed text
character_list = util.Pickle_read(cf.get('Pickle', 'path'), cf.get('Pickle', 'char_list'))

# Run the model generator on the given list if needed
if run_model_generator:
    # Create a word2vec model from the provided character list
    word2vec_creator = Word_vector_creator(character_list, cf.getint('Word2Vec', 'vector_size'), cf.getint('Word2Vec', 'window_size'))
    util.Pickle_write(cf.get('Pickle', 'path'), cf.get('Pickle', 'word2vec_model'), word2vec_creator.model)

# Load the saved/created model
word2vec_model = util.Pickle_read(cf.get('Pickle', 'path'), cf.get('Pickle', 'word2vec_model'))

if run_pedecerto:
    # Connect the Pedecerto output to this model
    line = -1 # 97 has lots of elision. 
    # Now call the parser and save the dataframe it creates
    parse = Pedecerto_parser(cf.get('Pedecerto', 'path_texts'), line)  
    parse.df.to_csv(cf.get('Pickle', 'pedecerto_df'), index = False, header=True)

pedecerto_df = pd.read_csv(cf.get('Pickle', 'pedecerto_df'), sep=',')

df = pedecerto_df
# Add syllable vectors to the dataframe using the word2vec model
df['vector'] = df['syllable']
# TODO: this might be in a try, although all syllables are present
for i in range(len(df)):
    syllable = df["syllable"][i]
    df["vector"][i] = word2vec_model.wv[syllable]
    # print(type(word2vec_model.wv[syllable]))

print(df)

num_lines = df["line"].max()
print(num_lines)
# max_value = column.max()

# Loop through the the lines and their vectors
for i in range(num_lines):
    # print(i)

    temp = df[df["line"] == i+1]
    
    
    
    exit(0)
    
    print(temp)

counter = 0

# for i in range(len(df)):
#     counter += 1
#     print(df["vector"][i], df["length"][i])

# while counter < 20:
#     counter += 1
#     print('[0]', 0)

temp = np.zeros(cf.getint('Word2Vec', 'vector_size'))
print(temp)
# print(type(temp))

# Now replace encoding by short and long
# df['length'] = np.where(df['length'] == 'A', 1, df['length'])
# df['length'] = np.where(df['length'] == 'T', 1, df['length'])
# df['length'] = np.where(df['length'] == 'b', 0, df['length'])
# df['length'] = np.where(df['length'] == 'c', 0, df['length'])

# print(df)
