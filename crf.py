# Basic imports
import configparser
import pandas as pd
import numpy as np
import collections
from progress.bar import Bar
import os
import matplotlib.pyplot as plt

# CRF specific imports
import scipy.stats
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
import sklearn_crfsuite
from sklearn_crfsuite import metrics as crf_metrics

# Class imports
import utilities as util
from experiments import CRF_experiments

class CRF_sequence_labeling:
    # feature-based sequence labelling: conditional random fields

    perform_pedecerto_conversion = False
    perform_convert_text_to_feature_sets = False
    perform_kfold = False
    perform_grid_search = False
    perform_fit_model = False
    perform_prediction_df = False
    perform_experiments = False
    custom_predict = True

    labels = ['short', 'long', 'elision']
    CONSONANTS = 'bcdfghjklmnpqrstvwxz'

    def __init__(self):

        # Read all the data we will be needing. The syllable_label_list contains a list of the used texts in [(syl, lbl), (syl,lbl), ...] format.
        crf_df = util.Pickle_read(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'crf_df'))
        # Load training and test set: X contains syllables and their features, y contains only scansion labels per line
        X = util.Pickle_read(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'crf_X'))
        y = util.Pickle_read(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'crf_y'))
        # Load our latest CRF model
        crf_model = util.Pickle_read(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'crf_model'))

        if self.perform_pedecerto_conversion:
            # Converts the pedecerto dataframe to a syllable_label_list as required by the used CRF suite
            texts = util.Create_files_list('./pickle', 'syllable_label')
            crf_df = self.convert_pedecerto_to_crf_df(texts)
            util.Pickle_write(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'crf_df'), crf_df)

        if self.perform_convert_text_to_feature_sets:
            # Takes the syllable label list and adds features to each syllable that are relevant for scansion
            X, y = self.convert_text_to_feature_sets(crf_df)
            util.Pickle_write(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'crf_X'), X)
            util.Pickle_write(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'crf_y'), y)

        if self.perform_fit_model:
            # Fit the model if needed
            crf_model = self.fit_model(X, y)
            self.print_crf_items(crf_model)
            util.Pickle_write(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'crf_model'), crf_model)

        if self.perform_kfold:
            # Perform kfold to check if we don't have any overfitting
            result = self.kfold_model(crf_df, X, y, 5)
            util.Pickle_write(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'crf_kfold_result'), result)
            print(result)

        if self.custom_predict:
            # Predict a custom sentence. NB: this has to be syllabified by the user
            custom_sentence = "li to ra mul tum il le et ter ris iac ta tus et al to"
            custom_sentence = "ar ma vi rum que ca no troi ae qui pri mus ab or is"
            self.predict_custom_sentence(crf_model, custom_sentence)            

        if self.perform_grid_search:
            # Does what it says on the tin
            self.grid_search(X, y)

        if self.perform_prediction_df:
            # Creates a simple prediction dataframe used by the frontend to quickly load results
            self.create_prediction_df(X, y)

        if self.perform_experiments:
            self.run_experiments()

    def run_experiments(self):
        create_models = True
        
        crf_exp_bar_dict = {'exp1': {}, 'exp2': {}, 'exp3': {}, 'exp4': {}, 'exp5': {}}

        # Experiment 1: Create model on Virgil, test on Virgil
        if create_models:
            texts = ['syllable_label_VERG-aene.xml.pickle']
            crf_df = self.convert_pedecerto_to_crf_df(texts)
            X, y = self.convert_text_to_feature_sets(crf_df)
            util.Pickle_write(util.cf.get('Pickle', 'path'), 'crf_exp1_X.pickle', X)
            util.Pickle_write(util.cf.get('Pickle', 'path'), 'crf_exp1_y.pickle', y)
        else:
            X = util.Pickle_read(util.cf.get('Pickle', 'path'), 'crf_exp1_X.pickle')
            y = util.Pickle_read(util.cf.get('Pickle', 'path'), 'crf_exp1_y.pickle')
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
        crf_model = self.fit_model(X_train, y_train)
        result = self.predict_model(crf_model, X_test, y_test)
        print('exp1')
       
        crf_exp_bar_dict['exp1'] = {'short_precision': result['short']['precision'],
                                    'short_recall': result['short']['recall'],
                                    'long_precision': result['long']['precision'],
                                    'long_recall': result['long']['recall'],
                                    'elision_precision': result['elision']['precision'],
                                    'elision_recall': result['elision']['recall'],
                                   }

        # Experiment 4: Test Virgil on Hercules Furens
        herc_df = pd.read_csv('HercFur.csv')  
        util.Pickle_write(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'test'), herc_df)
        herc_df = self.convert_pedecerto_to_crf_df(['test.pickle'])
        X_herc, y_herc = self.convert_text_to_feature_sets(herc_df)
        result = self.predict_model(crf_model, X_herc, y_herc)
        print('exp4')
        
        crf_exp_bar_dict['exp4'] = {'short_precision': result['short']['precision'],
                                    'short_recall': result['short']['recall'],
                                    'long_precision': result['long']['precision'],
                                    'long_recall': result['long']['recall'],
                                    'elision_precision': result['elision']['precision'],
                                    'elision_recall': result['elision']['recall'],
                                   }
        

        # Create model on Virgil, Ovid, Iuvenal and Lucretius, test on Aeneid
        texts = util.Create_files_list('./pickle', 'syllable_label')
        crf_df = self.convert_pedecerto_to_crf_df(texts)
        X, y = self.convert_text_to_feature_sets(crf_df)
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
        crf_model = self.fit_model(X_train, y_train)
        result = self.predict_model(crf_model, X_test, y_test)
        print('exp2')

        crf_exp_bar_dict['exp2'] = {'short_precision': result['short']['precision'],
                                    'short_recall': result['short']['recall'],
                                    'long_precision': result['long']['precision'],
                                    'long_recall': result['long']['recall'],
                                    'elision_precision': result['elision']['precision'],
                                    'elision_recall': result['elision']['recall'],
                                   }

        # Create model on Virgil, Ovid, Iuvenal and Lucreatius, test on all
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
        crf_model = self.fit_model(X_train, y_train)
        result = self.predict_model(crf_model, X_test, y_test)
        print('exp3')
        
        crf_exp_bar_dict['exp3'] = {'short_precision': result['short']['precision'],
                                    'short_recall': result['short']['recall'],
                                    'long_precision': result['long']['precision'],
                                    'long_recall': result['long']['recall'],
                                    'elision_precision': result['elision']['precision'],
                                    'elision_recall': result['elision']['recall'],
                                   }

        util.Pickle_write(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'test'), crf_exp_bar_dict)

        # herc_df = pd.read_csv('HercFur.csv')  
        # crf_df = self.convert_pedecerto_to_crf_df(['test.pickle'])
        # X, y = self.convert_text_to_feature_sets(herc_df)
        result = self.predict_model(crf_model, X_herc, y_herc)
        print('exp5')
        crf_exp_bar_dict['exp5'] = {'short_precision': result['short']['precision'],
                                    'short_recall': result['short']['recall'],
                                    'long_precision': result['long']['precision'],
                                    'long_recall': result['long']['recall'],
                                    'elision_precision': result['elision']['precision'],
                                    'elision_recall': result['elision']['recall'],
                                   }

        pd.DataFrame(crf_exp_bar_dict).T.plot(kind='bar')
        plt.legend(loc='lower left')
        plt.ylim([0.5, 1])
        plt.savefig('./result.png')
        plt.show()

    #############
    # FUNCTIONS #
    #############
    def kfold_model(self, crf_df, X, y, splits):
        if util.cf.get('Util', 'verbose'): print('Predicting the model')

        report_list = []

        # Convert the list of numpy arrays to a numpy array with numpy arrays
        X = np.array(X, dtype=object)
        y = np.array(y, dtype=object)
        kf = KFold(n_splits=splits, shuffle=True, random_state=42)

        for train_index, test_index in kf.split(crf_df):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            crf = self.fit_model(X_train, y_train)

            metrics_report = self.predict_model(crf, X_test, y_test)

            report_list.append(metrics_report)

        result = self.merge_kfold_reports(report_list)

        return result

    def predict_model(self, crf, X_test, y_test):
        y_pred = crf.predict(X_test)

        sorted_labels = sorted(
            self.labels,
            key=lambda name: (name[1:], name[0])
        )
        metrics_report = crf_metrics.flat_classification_report(
            y_test, y_pred, labels=sorted_labels, output_dict=True#, digits=3
        )
        return metrics_report

    def convert_text_to_feature_sets(self, syllable_label_list):
        if int(util.cf.get('Util', 'verbose')): print('Creating the X and y sets')

        X = [self.sentence_to_features(s) for s in syllable_label_list]
        y = [self.sentence_to_labels(s) for s in syllable_label_list]

        return X, y

    def fit_model(self, X, y) -> object:
        if int(util.cf.get('Util', 'verbose')): print('Fitting the model')

        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        crf.fit(X, y)

        return crf

    def predict_custom_sentence(self, crf_model, custom_sentence):
        # Turn the sentence into the format requested by the add_word_features function
        custom_sentence = custom_sentence.split()
        dummy_list = [0] * len(custom_sentence)
        combined_list = [(custom_sentence[i], dummy_list[i]) for i in range(0, len(custom_sentence))]
        custom_sentence_features = [self.add_word_features(combined_list, i) for i in range(len(combined_list))]

        # Print the confidence for every syllable
        marginals = crf_model.predict_marginals_single(custom_sentence_features)
        marginals = [{k: round(v, 4) for k, v in marginals.items()} for marginals in marginals]
        print(marginals)
        # Print the scansion for the entire sentence
        scansion = crf_model.predict_single(custom_sentence_features)
        print(scansion)
        # Below the reason for this scansion is saved
        # For every feature dictionary that is created per syllable.
        for idx, syllable_feature_dict in enumerate(custom_sentence_features):

            print('\nWe are now looking at syllable "{}".'.format(custom_sentence[idx]))
            print('I scanned it as being "{}"'.format(scansion[idx]))
            print('My confidence is {}:'.format(marginals[idx]))
            print('Below are the reasons why:')

            # Check if its features can be found in the big state_features dictionary.
            for key in syllable_feature_dict:
                my_string = str(key) + ':' + str(syllable_feature_dict[key])

                # If it is found, print the reasoning so we can examine it.
                for k, v in crf_model.state_features_.items():
                    if my_string in k:
                        print(k, v)

    def print_state_features(self, state_features):
        for (attr, label), weight in state_features:
            print("%0.6f %-8s %s" % (weight, label, attr))

    def add_word_features(self, sent, i):
        word = sent[i][0]

        # First, create features about the current word
        features = {
            'bias': 1.0,
            '0:last_3_char': word[-3:], # Take last 3 characters
            '0:last_2_char': word[-2:], # Take last 2 characters
            '0:position': i+1, #Note the position in the sentence
            '0:last_1_char_cons': self.str_is_consonants(word, slice(-1, None, None)),
            '0:last_2_char_cons': self.str_is_consonants(word, slice(-2, None, None)),
        }

        # Check if we are at the beginning of the sentence
        if i == 0:
            features['BOS'] = True # This should always be long

        # Gather features from the previous word
        if i > 0:
            previous_word = sent[i-1][0]
            features.update({
                '-1:word': previous_word,
                '-1:last_1_char': previous_word[-1:],
                '-1:last_2_char': previous_word[-2:],
            })

        # Gather features from the next word
        if i < len(sent)-1:
            next_word = sent[i+1][0]
            features.update({
                '+1:word': next_word,
                '+1:first_1_char': next_word[:1],
                '+1:first_2_char': next_word[:2],
                '+1:first_1_char_cons': self.str_is_consonants(next_word, slice(None, 2, None)),
                '+1:first_2_char_cons': self.str_is_consonants(next_word, slice(None, 2, None)),
            })
        else:
            features['EOS'] = True # This will be an anceps

        return features

    def sentence_to_features(self, sent):
        return [self.add_word_features(sent, i) for i in range(len(sent))]

    def sentence_to_labels(self, sent):
        return [label for token, label in sent]

    def sentence_to_tokens(self, sent):
        return [token for token, label in sent]

    def str_is_consonants(self, word, slicer) -> bool:
        return all([char in self.CONSONANTS for char in word[slicer]])

    def convert_pedecerto_to_crf_df(self, texts) -> list:
        if int(util.cf.get('Util', 'verbose')): print('Creating the sentence list')
        # Find all our texts and convert them to a crf dataframe
        # This will put the entire text into a list. In this list, another list is dedicated per sentence.
        # Each sentence list has tuples consisting of a syllable and its length (label).
        
        # Create a list to store all texts in
        all_sentences_list = []

        for text in texts:
            df = util.Pickle_read(util.cf.get('Pickle', 'path'), text)
            # Convert the integer labels to string labels 
            df = self.convert_syllable_labels(df)

            for title_index in Bar('Converting Pedecerto to CRF').iter(range(df['title'].max())):
                # Get only lines from this book
                title_df = df.loc[df['title'] == title_index + 1]
                # Per book, process the lines
                for line_index in range(title_df['line'].max()):
                    line_df = title_df[title_df["line"] == line_index + 1]

                    length_list = line_df['length'].to_numpy()
                    syllable_list = line_df['syllable'].to_numpy()
                    # join them into 2d array and transpose it to get the correct crf format:
                    combined_list = np.array((syllable_list,length_list)).T
                    # Append all to the list which we will return later
                    all_sentences_list.append(combined_list)

        return all_sentences_list
    
    def convert_syllable_labels(self, df):
        # Convert the labels from int to str
        df['length'] = np.where(df['length'] == 0, 'short', df['length'])
        df['length'] = np.where(df['length'] == 1, 'long', df['length'])
        df['length'] = np.where(df['length'] == 2, 'elision', df['length'])
        return df

    def print_crf_items(self, crf):
        print("Top positive:")
        self.print_state_features(collections.Counter(crf.state_features_).most_common(30))
        print("\nTop negative:")
        self.print_state_features(collections.Counter(crf.state_features_).most_common()[-30:])

    def merge_kfold_reports(self, report_list):
        result_dict = {
            'short': {'precision':0, 'recall':0, 'f1-score':0, 'support':0},
            'elision': {'precision':0, 'recall':0, 'f1-score':0, 'support':0},
            'long': {'precision':0, 'recall':0, 'f1-score':0, 'support':0},
            'weighted avg': {'precision':0, 'recall':0, 'f1-score':0, 'support':0},
        }

        keys = ['long', 'short', 'elision', 'weighted avg']

        # Merge the reports one by one
        for current_dict in report_list:
            for key in keys:
                result_dict[key] = self.merge_dicts(result_dict[key], current_dict[key])

        # Now divide all values by the number of reports that came in
        for dict in result_dict:
            for key in result_dict[dict]:
                result_dict[dict][key] /= len(report_list)

        return pd.DataFrame(result_dict).T

    def merge_dicts(self, dict1, dict2):
        return {k: dict1.get(k, 0) + dict2.get(k, 0) for k in dict1.keys() | dict2.keys()}

    def grid_search(self, X, y):

        if int(util.cf.get('Util', 'verbose')): print('Starting Gridsearch')
        X_train = X[:9000]
        y_train = y[:9000]
        X_test = X[9001:]
        y_test = y[9001:]

        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            max_iterations=100,
            all_possible_transitions=True
        )
        params_space = {
            'c1': scipy.stats.expon(scale=0.5),
            'c2': scipy.stats.expon(scale=0.05),
        }
        # use the same metric for evaluation
        f1_scorer = metrics.make_scorer(metrics.flat_f1_score,
                                average='weighted', labels=self.labels)

        # search
        rs = RandomizedSearchCV(crf, params_space,
                                cv=3,
                                verbose=1,
                                n_jobs=-1,
                                n_iter=50,
                                scoring=f1_scorer)
        rs.fit(X_train, y_train)

        print('best params:', rs.best_params_)
        print('best CV score:', rs.best_score_)
        print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))

        sorted_labels = sorted(
            self.labels,
            key=lambda name: (name[1:], name[0])
        )

        crf = rs.best_estimator_
        y_pred = crf.predict(X_test)
        print(metrics.flat_classification_report(
            y_test, y_pred, labels=sorted_labels, digits=3
        ))

        util.Pickle_write(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'seq_lab_rs'), rs)


    def create_prediction_df(self, X, y):
        # Creates a dataframe with predictions. Used by OSCC (for now)
        df = util.Pickle_read(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'flattened_vectors'))
        crf = util.Pickle_read(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'crf_model'))

        yhat = crf.predict(X)

        column_names = ["predicted", "expected"]
        new_df = pd.DataFrame(columns = column_names)

        for i in Bar('Processing').iter(range(len(y))):
            new_line = {'expected': y[i], 'predicted': yhat[i]}
            new_df = new_df.append(new_line, ignore_index=True)

        book_line_df = df[['book','line', 'syllable']]

        prediction_df = pd.concat([book_line_df, new_df], axis=1, join='inner')

        print(prediction_df)

        util.Pickle_write(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'seqlab_prediction_df'), prediction_df)


crf = CRF_sequence_labeling()
