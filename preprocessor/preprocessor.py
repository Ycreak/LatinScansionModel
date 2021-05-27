import re

# CLTK related
import cltk
from cltk.ir.query import search_corpus
from cltk.corpus.utils.importer import CorpusImporter
from cltk.corpus.readers import get_corpus_reader
from cltk.stem.latin.syllabifier import Syllabifier

import utilities as util

class Text_preprocessor:

    def __init__(self, text):


        corpus_importer = CorpusImporter('latin')
        corpus_importer.import_corpus('latin_text_perseus')
        # corpus_importer.import_corpus('latin_models_cltk')
        

        word_list = self.Get_word_list(text)

        # Clean the text
        word_list = self.Remove_numbers(word_list)
        word_list = self.Remove_element_from_list(word_list, '')
        word_list = self.Lowercase_list(word_list)

        word_list = self.Syllabify_list(word_list)

        self.character_list = word_list

    def Syllabify_list(self, given_list):
        syllabifier = Syllabifier()
        new_list = []
        
        for word in given_list:
            new_list.extend(syllabifier.syllabify(word))

        return new_list

    def Lowercase_list(self, given_list):
        return list(map(lambda x: x.lower(), given_list))

    def Remove_element_from_list(self, given_list, element):
        return list(filter(lambda x: x != element, given_list))

    def Get_word_list(self, text):
        """Reads the given texts and returns its words in a list

        Args:
            text (string): of the cltk json file

        Returns:
            list: of words
        """    
        reader = get_corpus_reader( corpus_name = 'latin_text_perseus', language = 'latin')
        docs = list(reader.docs())
        reader._fileids = [text]
        words = list(reader.words())

        return words

    # Strips the accents from the text for easier searching the text.
    # Accepts string with accentuation, returns string without.
    def strip_accents(self, s):
        return ''.join(c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')

    # Removes all the numbers and punctuation marks from a given list.
    # Returns list without these.
    def Remove_numbers(self, list): 
        pattern = '[0-9]|[^\w]'
        list = [re.sub(pattern, '', i) for i in list] 
        return list

    # Lemmatizes a given list. Returns a list with lemmatized words.
    def lemmatizeList(self, list):
        tagger = POSTag('greek')

        lemmatizer = LemmaReplacer('greek')
        lemmWords = lemmatizer.lemmatize(list)

        # Remove Stopwords and numbers and lowercases all words.
        lemmWords = [w.lower() for w in lemmWords if not w in STOPS_LIST]
        lemmWords = removeNumbers(lemmWords)

        return lemmWords    