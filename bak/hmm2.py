import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pprint import pprint 

import utilities as util
import configparser

from progress.bar import Bar

class Hidden_markov_model:

    cf = configparser.ConfigParser()
    cf.read("config.ini")

    pedecerto_df = util.Pickle_read(cf.get('Pickle', 'path'), cf.get('Pickle', 'pedecerto_df'))
    label_list = util.Pickle_read(cf.get('Pickle', 'path'), cf.get('Pickle', 'label_list'))
    
    def __init__(self):

        self.pedecerto_df = self.pedecerto_df_labels_to_str(self.pedecerto_df)        

        # create state space and initial state probabilities
        # label_list = create_sentence_list()

        # create state space and initial state probabilities
        hidden_states = ['long', 'short', 'elision']

        # create hidden transition matrix
        # a or alpha 
        #   = transition probability matrix of changing states given a state
        # matrix is size (M x M) where M is number of states
        # a_df = create_hidden_transition_matrix_alpha(hidden_states)
        a_df = util.Pickle_read(self.cf.get('Pickle', 'path'), self.cf.get('Pickle', 'hmm_a'))
        print(a_df)

        # create matrix of observation (emission) probabilities
        # b or beta = observation probabilities given state
        # matrix is size (M x O) where M is number of states 
        # and O is number of different possible observations
        unique_syllables = sorted(set(pedecerto_df['syllable'].tolist()))
        observable_states = unique_syllables

        # b_df = create_hidden_transition_matrix_beta(observable_states, hidden_states)
        b_df = util.Pickle_read(self.cf.get('Pickle', 'path'), self.cf.get('Pickle', 'hmm_b'))
        print(b_df)

        # observation sequence of dog's behaviors
        # observations are encoded numerically
        custom_sentence = "ar ma vi rum que ca no troi ae qui pri mus ab or is"
        custom_sentence = "li to ra mul tum il le et ter ris jac ta tus et al to"

        sentence_array = np.array([])

        for syllable in custom_sentence.split():
            sentence_array = np.append(sentence_array, observable_states.index(syllable))

        obs = sentence_array.astype(int)

        obs_map = {}
        for state in observable_states:
            # print(state, observable_states.index(state))
            obs_map[state] = observable_states.index(state)

        inv_obs_map = dict((v,k) for k, v in obs_map.items())
        obs_seq = [inv_obs_map[v] for v in list(obs)]

        # Sequence of overservations (and their code)
        print( pd.DataFrame(np.column_stack([obs, obs_seq]), 
                        columns=['Obs_code', 'Obs_seq']) )


        pi = get_label_probabilities()

        a = a_df.values
        b = b_df.values

        path, delta, phi = self.viterbi(pi, a, b, obs)
        # print('\nsingle best state path: \n', path)
        # print('delta:\n', delta)
        # print('phi:\n', phi)

        # exit(0)

        state_map = {0:'long', 1:'short',2:'elision'}
        state_path = [state_map[v] for v in path]

        print((pd.DataFrame()
        .assign(Observation=obs_seq)
        .assign(Best_Path=state_path)))


    def pedecerto_df_labels_to_str(df):
        df['length'] = np.where(df['length'] == 0, 'short', df['length'])
        df['length'] = np.where(df['length'] == 1, 'long', df['length'])
        df['length'] = np.where(df['length'] == 2, 'elision', df['length'])
        
        return df
    
    
    # define Viterbi algorithm for shortest path
    # code adapted from Stephen Marsland's, Machine Learning An Algorthmic Perspective, Vol. 2
    # https://github.com/alexsosn/MarslandMLAlgo/blob/master/Ch16/HMM.py
    def viterbi(self, pi, a, b, obs):
        
        nStates = np.shape(b)[0]
        T = np.shape(obs)[0]
        
        # init blank path
        path = path = np.zeros(T,dtype=int)
        # delta --> highest probability of any path that reaches state i
        delta = np.zeros((nStates, T))
        # phi --> argmax by time step for each state
        phi = np.zeros((nStates, T))
        
        # init delta and phi 
        delta[:, 0] = pi * b[:, obs[0]]
        phi[:, 0] = 0

        print('\nStart Walk Forward\n')    
        # the forward algorithm extension
        for t in range(1, T):
            for s in range(nStates):
                delta[s, t] = np.max(delta[:, t-1] * a[:, s]) * b[s, obs[t]] 
                phi[s, t] = np.argmax(delta[:, t-1] * a[:, s])
                print('s={s} and t={t}: phi[{s}, {t}] = {phi}'.format(s=s, t=t, phi=phi[s, t]))
        
        # find optimal path
        print('-'*50)
        print('Start Backtrace\n')
        path[T-1] = np.argmax(delta[:, T-1])
        for t in range(T-2, -1, -1):
            path[t] = phi[path[t+1], [t+1]]
            print('path[{}] = {}'.format(t, path[t]))
            
        return path, delta, phi

    def create_sentence_list(self) -> list:
        
        df = util.Pickle_read(self.cf.get('Pickle', 'path'), self.cf.get('Pickle', 'pedecerto_df'))
        # Entire Aneid is put into a list. In this list, a list is dedicated per sentence.
        # Each sentence list has tuples consisting of a syllable and its length.
        
        # Convert the labels from int to str
        df['length'] = np.where(df['length'] == 0, 'short', df['length'])
        df['length'] = np.where(df['length'] == 1, 'long', df['length'])
        df['length'] = np.where(df['length'] == 2, 'elision', df['length'])

        all_sentences_list = []

        # Get number of books to process
        num_books = df['book'].max()

        # for i in range(num_books):
        for i in Bar('Processing').iter(range(num_books)):
            # Get only lines from this book
            current_book = i + 1
            book_df = df.loc[df['book'] == current_book]

            num_lines = book_df['line'].max()

            for j in range(num_lines):
                current_line = j + 1

                filtered_df = book_df[book_df["line"] == current_line]

                length_list = filtered_df['length'].tolist()
                # syllable_list = filtered_df['syllable'].tolist()

                # combined_list = [(syllable_list[i], length_list[i]) for i in range(0, len(length_list))]

                all_sentences_list.append(length_list)

        util.Pickle_write(self.cf.get('Pickle', 'path'), self.cf.get('Pickle', 'label_list'), all_sentences_list)

        return all_sentences_list

    def create_hidden_transition_matrix_alpha(self, hidden_states):
    # Now we are going to fill the hidden transition matrix
        a_df = pd.DataFrame(columns=hidden_states, index=hidden_states)

        ll = 0
        ls = 0
        le = 0
        sl = 0
        ss = 0
        se = 0
        el = 0
        es = 0
        ee = 0

        total_count = 0

        for sentence in label_list:
            
            syllable_count = len(sentence)
            
            for idx, syllable in enumerate(sentence):

                if idx+1 < syllable_count:

                    item1 = sentence[idx]
                    item2 = sentence[idx+1]
            
                    if item1 == 'long' and item2 == 'long':ll +=1
                    elif item1 == 'long' and item2 == 'short': ls +=1
                    elif item1 == 'long' and item2 == 'elision': le +=1
                    elif item1 == 'short' and item2 == 'long': sl +=1
                    elif item1 == 'short' and item2 == 'short': ss +=1
                    elif item1 == 'short' and item2 == 'elision': se +=1
                    elif item1 == 'elision' and item2 == 'long': el +=1
                    elif item1 == 'elision' and item2 == 'short': es +=1
                    elif item1 == 'elision' and item2 == 'elision': ee +=1
                    else:
                        raise Exception('unknown transition found')

                else:
                    break

            total_count += syllable_count -1

            # print(syllable_count)
            # exit(0)

        prob_ll = ll/total_count
        prob_ls = ls/total_count
        prob_le = le/total_count
        prob_sl = sl/total_count
        prob_ss = ss/total_count
        prob_se = se/total_count
        prob_el = el/total_count 
        prob_es = es/total_count
        prob_ee = ee/total_count


        a_df.loc[hidden_states[0]] = [prob_ll, prob_ls, prob_le]
        a_df.loc[hidden_states[1]] = [prob_sl, prob_ss, prob_se]
        a_df.loc[hidden_states[2]] = [prob_el, prob_es, prob_ee]

        util.Pickle_write(self.cf.get('Pickle', 'path'), self.cf.get('Pickle', 'hmm_a'), a_df)

        return a_df

    def create_hidden_transition_matrix_beta(self, observable_states, hidden_states):

        b_df = pd.DataFrame(columns=observable_states, index=hidden_states)

        total_syllable_count = len(pedecerto_df)

        pedecerto_df['length'] = np.where(pedecerto_df['length'] == 0, 'short', pedecerto_df['length'])
        pedecerto_df['length'] = np.where(pedecerto_df['length'] == 1, 'long', pedecerto_df['length'])
        pedecerto_df['length'] = np.where(pedecerto_df['length'] == 2, 'elision', pedecerto_df['length'])

        for syllable in unique_syllables:
            filtered_df = pedecerto_df[pedecerto_df["syllable"] == syllable]
            
            filter = filtered_df['length'].value_counts()

            try:    
                b_df.at['long',syllable]    =filter['long']/total_syllable_count
            except:
                pass
            try:
                b_df.at['short',syllable]   =filter['short']/total_syllable_count
            except:
                pass

            try:
                b_df.at['elision',syllable] =filter['elision']/total_syllable_count
            except:
                pass

        b_df = b_df.fillna(0)
        util.Pickle_write(self.cf.get('Pickle', 'path'), self.cf.get('Pickle', 'hmm_b'), b_df)

        return b_df

    def get_label_probabilities(self):
        # print(pedecerto_df)

        pedecerto_df['length'] = np.where(pedecerto_df['length'] == 0, 'short', pedecerto_df['length'])
        pedecerto_df['length'] = np.where(pedecerto_df['length'] == 1, 'long', pedecerto_df['length'])
        pedecerto_df['length'] = np.where(pedecerto_df['length'] == 2, 'elision', pedecerto_df['length'])

        filter = pedecerto_df['length'].value_counts()

        # print(filter)

        long = filter['long']/len(pedecerto_df)
        short = filter['short']/len(pedecerto_df)
        elision = filter['elision']/len(pedecerto_df)

        # Return the probabilities of each hidden state
        return [long, short, elision]

