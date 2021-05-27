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

# Class Imports
from word2vec.word2vec import Word_vector_creator 
from preprocessor.preprocessor import Text_preprocessor 
import utilities as util

########
# MAIN #
########

# Temp params
run_preprocessor = False
run_model_generator = False

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


exit(0)

# Import files
import parser
import utilities
import numpy as np

# Import libraries
import numpy as np

utilities = utilities.Utility()

# Provide the folder with Pedecerto XML files here. These will be put in a pandas dataframe
path = './texts'
# Optionally, provide a single line for testing purposes
line = 0 # 97 has lots of elision. 
# Now call the parser and save the dataframe it creates
parse = parser.Parser(path, line)

df = parse.df
print(df)
exit(0)

# Now replace encoding by short and long
df['length'] = np.where(df['length'] == 'A', 1, df['length'])
df['length'] = np.where(df['length'] == 'T', 1, df['length'])
df['length'] = np.where(df['length'] == 'b', 0, df['length'])
df['length'] = np.where(df['length'] == 'c', 0, df['length'])

print(df)
