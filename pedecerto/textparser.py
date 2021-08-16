from bs4 import BeautifulSoup

import pandas as pd
import string
import os
import timeit
from progress.bar import Bar

import utilities as util

import pedecerto.rhyme as pedecerto

class Pedecerto_parser:
  """This class parses the Pedecerto XML into a dataframe which can be used for
  training models.
  """  

  df = pd.DataFrame()
  
  # utilities = utilities.Utility()
  # constants = ScansionConstants()
  
  def __init__(self, path, givenLine = -1):
    # Create pandas dataframe
    column_names = ["author", "text", "book", "line", "syllable", "length"]
    # column_names = ["author", "text", "line", "syllable", "foot", "feet_pos", "length", "word_boundary", "metrical_feature"]
    df = pd.DataFrame(columns = column_names) 
    
    # Add all entries to process to a list
    entries = util.Create_files_list(path, 'xml')
    # Process all entries added to the list
    for entry in entries:
      with open(path + entry) as fh:
        # Use beautiful soup to process the xml
        soupedEntry = BeautifulSoup(fh,"xml")
        # Retrieve the title and author from the xml file
        text_title = str(soupedEntry.title.string)
        author = str(soupedEntry.author.string)
        # Clean the lines (done by MQDQ)
        soupedEntry = util.clean(soupedEntry('line'))

        if givenLine == -1: #FIXME: deprecated
          # Do the entire folder
          # for line in range(len(soupedEntry)):
          for line in Bar('Processing {0}, {1}'.format(author, text_title)).iter(range(len(soupedEntry))):
            # progress_percentage = round(line / len(soupedEntry) * 100 ,2)
            # print('Progress on {0}, {1}: line {2} of {3} processed: {4}%'.format(self.author, self.title, line, len(soupedEntry), progress_percentage))

            book_title = int(soupedEntry[line].parent.get('title'))

            # Process the entry. It will append the line to the df
            line_df = self.Process_line(soupedEntry[line], book_title, text_title, author)
            df = df.append(line_df, ignore_index=True)
            # If I greatly improve my own code, am I a wizard, or a moron?
        
          print(df)

        else:
          # Process just the given line (testing purposes).
          # df = self.Process_line(soupedEntry[givenLine], df)
          pass
      # Store df for later use
      self.df = df #FIXME: better name. How shall we store and exchange between classes?

  def Process_line(self, givenLine, book_title, text_title, author):
    """Processes a given XML pedecerto line. Puts syllable and length in a dataframe.

    Args:
        givenLine (xml): pedecerto xml encoding of a line of poetry
        df (dataframe): to store the data in
        book_title (str): title of the current book (Book 1)
        text_title (str): title of the current text (Aeneid)
        author (str): name of the current author

    Returns:
        dataframe: with syllables and their lenght (and some other information)
    """      
    column_names = ["author", "text", "book", "line", "syllable", "length"]
    df = pd.DataFrame(columns = column_names)
    
    current_line = givenLine['name']

    # Parse every word and add its features
    for w in givenLine("word"):
      
      # Now for every word, syllabify it first
      word_syllable_list = pedecerto._syllabify_word(w)

      # And get its scansion
      scansion = w["sy"]

      # Check how many syllables we have according to pedecerto
      split_scansion = [scansion[i:i+2] for i in range(0, len(scansion), 2)] # per two characters

      # We use this to detect elision      
      number_of_scansions = len(split_scansion)

      for i in range(len(word_syllable_list)):
        # Now we loop through the syllable list of said word and extract features
        current_syllable = word_syllable_list[i].lower()
        
        # If we still have scansions available
        if number_of_scansions > 0:

          foot = split_scansion[i][0]
          feet_pos = split_scansion[i][1]

          # Interpret length based on pedecerto encoding
          if feet_pos == 'A':
            length = 1
          elif feet_pos == 'T':
            length = 1
          elif feet_pos == 'b':
            length = 0
          elif feet_pos == 'c':
            length = 0
          elif feet_pos == '':
            length = -1        

        # No scansions available? Elision. Denote with -1
        else:
          length = -1
          feet_pos = 'NA'
          foot = 'NA'

        # Keep track of performed operations
        number_of_scansions -= 1

        # Append to dataframe
        new_line = {'author': author, 'text': text_title, 'book': book_title, 'line': current_line, 'syllable': current_syllable, 'length': length}
        df = df.append(new_line, ignore_index=True)

    return df

# UNUSED FUNCTIONS (for now)
  # def AddFeature_Speech(self, df):
  #   df['liquids'] = 0
  #   df['nasals'] = 0
  #   df['fricatives'] = 0
  #   # df['clusterable'] = 0
  #   df['mutes'] = 0
  #   df['aspirate'] = 0
  #   df['doubled_consonant'] = 0

  #   df['char_first'] = 0
  #   df['char_second'] = 0
  #   df['char_ultima'] = 0
  #   df['char_penultima'] = 0

  #   for i in range(len(df)):
  #     if any(liquid in df["syllable"][i] for liquid in self.constants.LIQUIDS):
  #       df['liquids'][i] = 1
  #     if any(nasal in df["syllable"][i] for nasal in self.constants.NASALS):
  #       df['nasals'][i] = 1
  #     if any(fricative in df["syllable"][i] for fricative in self.constants.FRICATIVES):
  #       df['fricatives'][i] = 1
  #     if any(mute in df["syllable"][i] for mute in self.constants.MUTES):
  #       df['mutes'][i] = 1        
  #     if any(aspirate in df["syllable"][i] for aspirate in self.constants.ASPIRATES):
  #       df['mutes'][i] = 1    
      
  #     df = self.CheckConsonantStatus(df["syllable"][i], df, i)

  #   return df

  # def CheckConsonantStatus(self, string, df, i):
    
  #   char_first = string[0] 
  #   char_ultima = string[-1] 

  #   try:
  #     char_second = string[1]
  #     char_penultima = string[-2]
  #   except:
  #     char_second = '-'
  #     char_penultima = '-'
  #     print('String probably one character, continuing')

  #   if char_first in self.constants.CONSONANTS:
  #     print('first char is consonant')
  #     df['cons_first'][i] = 1

  #   if char_second in self.constants.CONSONANTS:
  #     print('second char is consonant')
  #     df['cons_second'][i] = 1

  #   if char_ultima in self.constants.CONSONANTS:
  #     print('last char is consonant')
  #     df['cons_ultima'][i] = 1

  #   if char_penultima in self.constants.CONSONANTS:
  #     print('second to last char is consonant')
  #     df['cons_penultima'][i] = 1

  #   return df


  # def AddFeature_Diphthong(self, df):
  #   """Adds a diphtong feature to the given dataframe based on the syllable column

  #   Args:
  #       df (dataframe): does what it says on the tin

  #   Returns:
  #       df: dataframe with diphtong column appended
  #   """    
  #   # Initialise diphthong column to 0.
  #   df['diphtong'] = 0
    
  #   for i in range(len(df)):
  #     if any(diphtong in df["syllable"][i] for diphtong in self.constants.DIPTHONGS):
  #       df['diphtong'][i] = 1
        
  #   return df



