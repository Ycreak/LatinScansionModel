import configparser
import pandas as pd
import numpy as np
from progress.bar import Bar

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

    def __init__(self, df):
        # Read the config file for later use
        self.cf = configparser.ConfigParser()
        self.cf.read("config.ini") 

        add_padding = False
        make_neural_readable = False
        flatten_vector = False
        create_model = False

        # This functions add padding to every line
        if add_padding:
            print('Adding padding')
            df = self.Add_padding(df)
            util.Pickle_write(self.cf.get('Pickle', 'path'), self.cf.get('Pickle', 'padded_set'), df)

        df = util.Pickle_read(self.cf.get('Pickle', 'path'), self.cf.get('Pickle', 'padded_set'))

        if make_neural_readable:
            # This abomination puts each line in a single dataframe row
            # Now the vectors and lengths are both put into their own list, divided into two columns in the df.
            print("Creating vector-target dataframe")
            df = self.Turn_df_into_neural_readable(df)
            util.Pickle_write(self.cf.get('Pickle', 'path'), self.cf.get('Pickle', 'neural_readable'), df)
        df = util.Pickle_read(self.cf.get('Pickle', 'path'), self.cf.get('Pickle', 'neural_readable'))

        
        if flatten_vector:
            df = self.Flatten_vector(df)
            util.Pickle_write(self.cf.get('Pickle', 'path'), self.cf.get('Pickle', 'flattened_vectors'), df)
        df = util.Pickle_read(self.cf.get('Pickle', 'path'), self.cf.get('Pickle', 'flattened_vectors'))

        ####
        # TODO: Philippe plz continue here
        ####

        # Turn df into X and y for neural network
        print('Creating X and y')
        X, y = self.Create_X_y(df, use_file=False)

        print("Training data: shape={}".format(X.shape))
        print("Training target data: shape={}".format(y.shape))
      
        if create_model:
            # Specify the input dimension of the network
            _input_dim = int(self.cf.get('NeuralNetwork', 'max_length')) * int(self.cf.get('Word2Vec', 'vector_size'))

            # Neural Network parameters
            _output_layer_size = int(self.cf.get('NeuralNetwork', 'max_length'))
            _epochs = int(self.cf.getint('NeuralNetwork', 'epochs'))
            _batch_size = int(self.cf.getint('NeuralNetwork', 'batch_size'))

            # Split test and train set
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

            # Scale data #FIXME: do we need a scaler? All values are between -1 and 1.
            # scaler = MinMaxScaler()
            # X_train = scaler.fit_transform(X_train)
            # X_test = scaler.transform(X_test)

            # Create the model #FIXME: should we put this inside a function?
            model = Sequential()
            model.add(Dense(12, input_dim=_input_dim, activation='relu'))
            model.add(Dense(8, activation='relu'))
            model.add(Dense(_output_layer_size, activation='sigmoid'))
            # compile the keras model
            model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
            # # Train
            history = model.fit(X_train, y_train, epochs=_epochs, batch_size=_batch_size, 
                                validation_data=(X_test, y_test), shuffle=True)

            _, train_accuracy = model.evaluate(X_train, y_train)
            _, test_accuracy = model.evaluate(X_test, y_test)

            print('Accuracy (training): %.2f' % (train_accuracy * 100))
            print('Accuracy (testing): %.2f' % (test_accuracy * 100))

            model.save('pickle/model')
            # util.Pickle_write(self.cf.get('Pickle', 'path'), self.cf.get('Pickle', 'neural_model'), history)

        # Test the model #FIXME: only one output?
        model = models.load_model('pickle/model')

        x_new = X[1:5]

        y_new = model.predict_classes(x_new)

        # print('{0} => {1} (expected {2})'.format((X[0], predictions[0], y[0])))
        for i in range(len(x_new)):
            # print("X=%s, Predicted=%s" % (x_new[i], y_new[i], y[i]))
            print("X={0}, Predicted={1}, Expected={2}".format(x_new[i], y_new[i], y[i]))



    def Flatten_vector(self, df):

        for i in range(len(df)): # For debugging    
            # print(df["vector"][i])
            
            array = np.array(df["vector"][i]).flatten().tolist()
            df["vector"][i] = array

        return df

    def Create_X_y(self, df, use_file=False):

        X = []
        y = []

        X = list()
        y = list()
        for _, row in df.iterrows():
            X.append(row['vector'])
            y.append(row['target'])
        
        # Here I learned that tensorflow errors are unreadable. Lest we forget.
        # ValueError: setting an array element with a sequence.

        X = np.array(X, dtype=np.float)
        y = np.array(y, dtype=np.float)

        return X, y

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
        # same_line = True
        zero_vector = np.zeros(self.cf.getint('Word2Vec', 'vector_size'))
        max_length_sentence = self.cf.getint('NeuralNetwork', 'max_length')

        # This is so very C++, it hurts. #FIXME: to fix this, we need a column BOOK or TEXT to distinguish between the different line 1 and line 2 etc.
        # However, this is not the same for every text, thus the reason for this approach.
        for i in Bar('Processing').iter(range(len(df))):
        # for i in range(len(df)): # For debugging    
            current_line = int(df["line"][i]) # Of course, pandas saves my integer as a string

            if current_line == previous_line:
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
            
        return new_df

    def Turn_df_into_neural_readable(self, df):
        column_names = ["vector", "target"]
        nn_df = pd.DataFrame(columns = column_names) 

        previous_line = 1
        same_line = True
        vector_list = []
        target_list = []

        # for i in range(len(df)):
        for i in Bar('Processing').iter(range(len(df))):
            # This is not DRY
            current_line = df["line"][i]

            if current_line == previous_line:
                same_line = True
            else:
                same_line = False

            if same_line:
                # We are working in the same line!                         
                vector_list.append(df['vector'][i]) # Create a list of the vectors
                target_list.append(df['length'][i]) # Create a list of the targets

            else:
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


        
