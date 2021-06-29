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
    prepare_for_network = True

    if add_padding:
        df = Add_padding(df, cf)

    df = pd.read_csv(cf.get('Pickle', 'training_set'), sep=',')



    if prepare_for_network:

        column_names = ["vector", "target"]
        new_df = pd.DataFrame(columns = column_names) 

        df = df.head(100)

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
                new_df = new_df.append(new_line, ignore_index=True)

                vector_list = []
                target_list = []

                vector_list.append(df['vector'][i])
                target_list.append(df['length'][i])

            previous_line = current_line

        print(new_df)
        new_df.to_csv(cf.get('Pickle', 'neural_set'), index = False, header=False)

    # Now let us get the network working
    # X = new_df.iloc[:,0:10]
    # y = new_df.iloc[:,10]

    y = new_df.pop('target')
    X = new_df

    #TODO: when converting X to a numpy object, it converts the vector list to a list([' object

    # y = y.to_numpy()
    # X = X.to_numpy()

    # print(X)

    # https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
    # # define the keras model
    # model = Sequential()
    # model.add(Dense(12, input_dim=20, activation='relu'))
    # model.add(Dense(8, activation='relu'))
    # model.add(Dense(20, activation='sigmoid'))

    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # # fit the keras model on the dataset
    # model.fit(X, y, epochs=150, batch_size=10)
    # # evaluate the keras model
    # _, accuracy = model.evaluate(X, y)
    # print('Accuracy: %.2f' % (accuracy*100))

    dataset = loadtxt('test.csv', delimiter=',')
    # split into input (X) and output (y) variables
    X = dataset[:,0:8]
    y = dataset[:,8]

    print(X)

    # dataset = np.loadtxt(new_df, delimiter=',')
    

    # X = dataset[:,0:20]
    # y = dataset[:,20]


        #     while counter < max_length_sentence:
        #         new_line = {'line': previous_line, 'syllable': 0, 'length': -100, 'vector': zero_vector}
        #         new_df = new_df.append(new_line, ignore_index=True)
        #         counter += 1

        #     # We created padding, now continue as normal
        #     counter = 0

        #     new_line = {'line': current_line, 'syllable': df["syllable"][i], 'length': df["length"][i], 'vector': df["vector"][i]}
        #     new_df = new_df.append(new_line, ignore_index=True)
        #     counter += 1
        
        # # Update current line
        # previous_line = current_line
    # print(df)
        