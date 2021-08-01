import configparser
import pandas as pd
import numpy as np

from numpy import loadtxt

from tensorflow.keras import models
from tensorflow.keras import layers
# from keras.models import Sequential
# from keras.layers import Dense

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import utilities as util

class Neural_network_handler:
    """This class handles everything neural network related.
    """  

    def __init__(self, use_file):
        # Read the config file for later use
        self.cf = configparser.ConfigParser()
        self.cf.read("config.ini") 

        X, y = nn.load_data(df, use_file=False) # Either by finalizing parsing or by files

        print("Training data: shape={}".format(X.shape))
        print("Training target data: shape={}".format(y.shape))

        exit(0)

        # Split test and train set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        # Scale data
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Create the model
        model = create_model(X_train, y_train)

        # # Train
        history = model.fit(X_train, y_train, epochs=self.cf.getint('NeuralNetwork', 'epochs'), batch_size=self.cf.getint('NeuralNetwork', 'batch_size'), 
            validation_data=(X_test, y_test), shuffle=True)

        _, train_accuracy = model.evaluate(X_train, y_train)
        _, test_accuracy = model.evaluate(X_test, y_test)

        print('Accuracy (training): %.2f' % (train_accuracy * 100))
        print('Accuracy (testing): %.2f' % (test_accuracy * 100))


    def Add_padding(self, df): 
        """Adds padding vectors to each sentence to give the neural network a constant length per sentence.
        For example, if a sentence has 12 syllables, 8 padding vectors will be added.

        Args:
            df (dataframe): contains sentences to be padded

        Returns:
            df: with padded sentences
        """    
        column_names = ["line", "syllable", "length", "vector"]
        new_df = pd.DataFrame(columns = column_names) 
        
        # Loop through the the lines and their vectors
        previous_line = 1
        counter = 0
        same_line = True
        zero_vector = np.zeros(self.cf.getint('Word2Vec', 'vector_size'))
        max_length_sentence = self.cf.getint('NeuralNetwork', 'max_length')

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
        
        # new_df.to_csv(self.cf.get('Pickle', 'training_set'), index = False, header=True)
        util.Pickle_write(cf.get('Pickle', 'path'), cf.get('Pickle', 'training_set'), new_df)

        return new_df

    def Turn_df_into_neural_readable(self, df):
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
                
                # print(type(df['vector'][i]))  FIXME: flatten list here?
                # # print(df['vector'][i].to_numpy())

                # # exit(0)
                # print(np.fromstring(df['vector'][i], dtype=float))

                # # print(df['vector'][i].flatten())
                # exit(0)
                        
                vector_list.append(df['vector'][i]) # Create a list of the vectors
                target_list.append(df['length'][i]) # Create a list of the targets

            else:
                # vector_list.extend(target_list) #FIXME: removed this issue.

                # print(vector_list.flatten())
                # exit(0)

                new_line = {'vector': vector_list, 'target': target_list}
                nn_df = nn_df.append(new_line, ignore_index=True)

                vector_list = []
                target_list = []

                vector_list.append(df['vector'][i])
                target_list.append(df['length'][i])

            previous_line = current_line

        return(nn_df)

    def load_data(self, df, use_file=False):
        if use_file:
            with open('./pickle/X.npy', 'rb') as f:
                X = np.load(f, allow_pickle=True)
            with open('./pickle/y.npy', 'rb') as f:
                y = np.load(f, allow_pickle=True)
            return X, y

        
        add_padding = False


        # This functions add padding to every line
        if add_padding:
            print('Adding padding')
            df = Add_padding(df)
        df = util.Pickle_read(cf.get('Pickle', 'path'), cf.get('Pickle', 'training_set'))
        # pd.read_csv(self.cf.get('Pickle', 'training_set'), sep=',')

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

    def create_model(self, X_train, y_train):
        """Create a neural network with two hidden layers,
             Dependent on the sizes of the training data

        Args:
            X_train (array): .
            y_train (array): .

        Returns:
            ...
        """    
        model = Sequential()
        model.add(Dense(self.cf.get('NeuralNetwork', 'hidden_size'), input_dim=X_train.shape[1], activation='relu'))
        model.add(Dense(self.cf.get('NeuralNetwork', 'hidden_size'), activation='relu'))
        model.add(Dense(y_train.shape[1], activation='sigmoid'))
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        return model


        
