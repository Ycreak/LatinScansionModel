import configparser
import pandas as pd
import numpy as np

from numpy import loadtxt

from tensorflow.keras import models
from tensorflow.keras import layers
# from keras.models import Sequential
# from keras.layers import Dense

import utilities as util

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

    previous_line = 1
    same_line = True
    vector_list = []
    target_list = []

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

        else:
            vector_list.extend(target_list)

            new_line = {'vector': vector_list, 'target': target_list}
            nn_df = nn_df.append(new_line, ignore_index=True)

            vector_list = []
            target_list = []

            vector_list.append(df['vector'][i])
            target_list.append(df['length'][i])

        previous_line = current_line

    return(nn_df)

def load_data(df, use_file=False):
    if use_file:
        with open('./pickle/X.npy', 'rb') as f:
            X = np.load(f, allow_pickle=True)
        with open('./pickle/y.npy', 'rb') as f:
            y = np.load(f, allow_pickle=True)
        return X, y
    
    # Read the config file for later use
    cf = configparser.ConfigParser()
    cf.read("config.ini")
    
    add_padding = False

    # This functions add padding to every line
    if add_padding:
        print('Adding padding')
        df = Add_padding(df, cf)
    df = pd.read_csv(cf.get('Pickle', 'training_set'), sep=',')

    # This abomination puts each line in a single dataframe row
    # Now the vectors and lengths are both put into their own list, divided into two columns in the df.

    print("Creating vector-target dataframe")
    df = Turn_df_into_neural_readable(df)

    y = list()
    for _, row in df.iterrows():
        y.append(row['target'])
    y = np.array(y, dtype=np.float)

    X = list()
    for _, row in df.iterrows():
        X.append(row['vector'])
    
    for i, s in enumerate(X):
        for i2, svalue in enumerate(s):
            if isinstance(svalue, str):
                X[i][i2] = svalue.replace("[", "")
                X[i][i2] = X[i][i2].replace("]", "")
                X[i][i2] = X[i][i2].replace("\'", "")
                X[i][i2] = X[i][i2].replace("\n", "")
                X[i][i2] = X[i][i2].replace("\\", "")
                X[i][i2] = X[i][i2].replace("n", "")
                X[i][i2] = X[i][i2].split()
                X[i][i2] = np.array([float(f) for f in X[i][i2]], dtype=np.float) # This is not the solution, we should not lose precision
            else:
                pass
        while not isinstance(X[i][-1], np.ndarray):
            X[i].pop() # remove targets (not needed anymore)

    X = np.array(X)
    with open('./pickle/X.npy', 'wb') as f:
        np.save(f, X)
    with open('./pickle/y.npy', 'wb') as f:
        np.save(f, y)
    return X, y



        
