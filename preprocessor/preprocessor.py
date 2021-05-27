# CLTK related
import cltk
from cltk.ir.query import search_corpus
from cltk.corpus.utils.importer import CorpusImporter
from cltk.corpus.readers import get_corpus_reader

import utilities as util

class Text_preprocessor:

    def __init__(self, text):


        corpus_importer = CorpusImporter('latin')
        corpus_importer.import_corpus('latin_text_perseus')
        # corpus_importer.import_corpus('latin_models_cltk')
        
        self.character_list = self.Get_word_list(text)

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

