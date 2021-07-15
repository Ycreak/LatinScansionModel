import configparser
import pandas as pd
import numpy as np

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

def Add_padding(df, cf):
    
    column_names = ["line", "syllable", "length", "vector"]
    new_df = pd.DataFrame(columns = column_names) 
    
    # Loop through the the lines and their vectors
    previous_line = 1
    counter = 0
    same_line = True
    zero_vector = np.zeros(cf.getint('Word2Vec', 'vector_size'))
    max_length_sentence = cf.getint('NeuralNetwork', 'max_length')

    # This is so very C++, it hurts. #FIXME: to fix this, we need a column BOOK or TEXT to distinguish between the different line 1 and line 2 etc.
    # However, this is not the same for every text, thus the reason for this approach.
    for i in range(len(df)):
        current_line = df["line"][i]

        if current_line == previous_line:
            same_line = True
        else:
            same_line = False

        if same_line:
            # We are working in the same line!            
            new_line = {'line': current_line, 'syllable': df["syllable"][i], 'length': df["length"][i], 'vector': df["vector"][i]}
            new_df = new_df.append(new_line, ignore_index=True)
            counter += 1

        else:
            while counter < max_length_sentence:
                new_line = {'line': previous_line, 'syllable': 0, 'length': -100, 'vector': zero_vector}
                new_df = new_df.append(new_line, ignore_index=True)
                counter += 1

            # We created padding, now continue as normal
            counter = 0

            new_line = {'line': current_line, 'syllable': df["syllable"][i], 'length': df["length"][i], 'vector': df["vector"][i]}
            new_df = new_df.append(new_line, ignore_index=True)
            counter += 1
        
        # Update current line
        previous_line = current_line
        
    print(new_df)    
    
    new_df.to_csv(cf.get('Pickle', 'training_set'), index = False, header=True)

    return new_df

def Turn_df_into_neural_readable(df):
    column_names = ["vector", "target"]
    nn_df = pd.DataFrame(columns = column_names) 

    # df = df.head(100)

    previous_line = 1
    counter = 0
    same_line = True
    vector_list = []
    target_list = []

    print(df)

    for i in range(len(df)):
        # This is not DRY
        current_line = df["line"][i]

        if current_line == previous_line:
            same_line = True
        else:
            same_line = False

        if same_line:
            # We are working in the same line!            
            vector_list.append(df['vector'][i])
            target_list.append(df['length'][i])

            # print(vector_list)

        else:
            # print(vector_list)

            vector_list.extend(target_list)
            # result_list = vector_list

            new_line = {'vector': vector_list, 'target': target_list}
            nn_df = nn_df.append(new_line, ignore_index=True)

            vector_list = []
            target_list = []

            vector_list.append(df['vector'][i])
            target_list.append(df['length'][i])

        previous_line = current_line


    return(nn_df)

def Start_neural_network(df):

    # Read the config file for later use
    cf = configparser.ConfigParser()
    cf.read("config.ini")
    
    add_padding = False
    prepare_for_network = True

    # This functions add padding to every line
    if add_padding:
        df = Add_padding(df, cf)

    df = pd.read_csv(cf.get('Pickle', 'training_set'), sep=',')

    # This abomination puts each line in a single dataframe row
    # Now the vectors and lengths are both put into their own list, divided into two columns in the df.
    if prepare_for_network:

        df = Turn_df_into_neural_readable(df)
        print(df)

        df.to_csv(cf.get('Pickle', 'neural_set'), index = False, header=False)




        