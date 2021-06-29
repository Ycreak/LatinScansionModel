import re

# CLTK related
import cltk
from cltk.ir.query import search_corpus
from cltk.corpus.utils.importer import CorpusImporter
from cltk.corpus.readers import get_corpus_reader
from cltk.stem.latin.syllabifier import Syllabifier

from pedecerto.rhyme import *

from bs4 import BeautifulSoup

import utilities as util

class Text_preprocessor:

    def __init__(self, text):

    # # Add all entries to process to a list
    # entries = self.CreateFilesList(path, 'xml')
    # # Process all entries added to the list
    # for entry in entries:
    #   with open(path + entry) as fh:
    #     # Use beautiful soup to process the xml
    #     soupedEntry = BeautifulSoup(fh,"xml")
    #     # Retrieve the title and author from the xml file
    #     self.title = soupedEntry.title.string
    #     self.author = soupedEntry.author.string
    #     # Clean the lines (done by MQDQ, don't know what it does exactly)
    #     soupedEntry = util.clean(soupedEntry('line'))
    #     if givenLine == -1:
    #       # Do the entire folder
    #       # for line in range(len(soupedEntry)):
    #       for line in range(2):
    #         print('Progress on', self.author, self.title, ':', round(line / len(soupedEntry) * 100, 2), "%")
    #         # Process the entry. It will append the line to the df
    #         self.df = self.ProcessLine(soupedEntry[line], self.df)
    #     else:
    #       # Process just the given line (testing purposes).
    #       self.df = self.ProcessLine(soupedEntry[givenLine], self.df)
    
    # Now add features to the dataframe
    # self.df = self.AddFeature_Diphthong(self.df)
    # self.df = self.AddFeature_Speech(self.df)
  
        word_list = []
        # TODO: not hard coded text
        with open('texts/VERG-aene.xml') as fh:
            # Use beautiful soup to process the xml
            soupedEntry = BeautifulSoup(fh,"xml")
            # Retrieve the title and author from the xml file
            soupedEntry = util.clean(soupedEntry('line'))
            # Do the entire folder
            for line in range(len(soupedEntry)):
                # print('Progress on', ':', round(line / len(soupedEntry) * 100, 2), "%")
                # Process the entry. It will append the line to the df
                # self.df = 
                word_list.extend(self.Syllabify_line(soupedEntry[line]))

        
        # Clean the text (already done by pedecerto)
        word_list = self.Remove_numbers(word_list)
        word_list = self.Remove_element_from_list(word_list, '')
        word_list = self.Lowercase_list(word_list)

        # print(word_list)

        self.character_list = word_list

    # Returns the dataframe appended
    def Syllabify_line(self, givenLine):

        all_syllables = syllabify_line(givenLine)
        
        # Flatten list (hack)
        all_syllables = [item for sublist in all_syllables for item in sublist]

        return all_syllables


    # Returns the dataframe appended
    def ProcessLine(self, givenLine):
        # syllabifier = Syllabifier()

        # all_syllables = syllabify_line(givenLine)
        # # Flatten list (hack)
        # all_syllables = [item for sublist in all_syllables for item in sublist]

        temp_list = []

        words = givenLine.find_all('word')

        for word in words:
            temp_list.append(word.string)

            # print(myWord)


        # print(words)
        return temp_list

        exit(0)

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