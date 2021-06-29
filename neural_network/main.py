import configparser
import pandas as pd
import numpy as np

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

def Start_neural_network(df):

    # Read the config file for later use
    cf = configparser.ConfigParser()
    cf.read("config.ini")
    
    add_padding = False

    if add_padding:
        df = Add_padding(df, cf)

    df = pd.read_csv(cf.get('Pickle', 'training_set'), sep=',')

    print(df.head(30))
        

    # for i in range(len(df)):
    #     counter += 1
    #     print(df["vector"][i], df["length"][i])

    # while counter < 20:
    #     counter += 1
    #     print('[0]', 0)

    # zero_vector = np.zeros(cf.getint('Word2Vec', 'vector_size'))
    # print(temp)
    # print(type(temp))