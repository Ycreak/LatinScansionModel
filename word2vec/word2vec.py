# Import gensim 
from gensim.models import Word2Vec

class Word_vector_creator:
    def __init__(self, list, vector_size, window_size):
        """This class expects a list of characters and converts the characters into a gensim model.
        The model is then saved to disk for later use.

        Args:
            list (list): of characters to be converted into a word2vec model

        Returns:
            object: gensim word2vec model
        """


        self.model = self.Generate_Word2Vec_model(list, vector_size, window_size)
                

    def Generate_Word2Vec_model(self, vector_list, vector_size, window_size):
        """Returns Word2Vec model generated from the given text-list

        Args:
            vectorList (list): of words of the text we want to create the model on
            size (int): of dimension
            window (int): size of window

        Returns:
            object: the word2vec model
        """    
        return Word2Vec([vector_list], vector_size=vector_size, window=window_size, min_count=1, workers=4)
