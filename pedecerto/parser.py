from bs4 import BeautifulSoup
# from cltk.stem.latin.syllabifier import Syllabifier
# from cltk.prosody.latin.syllabifier import Syllabifier

import pandas as pd
import string
import os

import utilities as util
# from bak.scansion_constants import ScansionConstants

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
    column_names = ["author", "text", "line", "syllable", "length"]
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
        self.title = soupedEntry.title.string
        self.author = soupedEntry.author.string
        # Clean the lines (done by MQDQ)
        soupedEntry = util.clean(soupedEntry('line'))

        if givenLine == -1:
          # Do the entire folder
          for line in range(len(soupedEntry)):
            print('Progress on', self.author, self.title, ':', round(line / len(soupedEntry) * 100, 2), "%")
            # Process the entry. It will append the line to the df
            df = self.Process_line(soupedEntry[line], df)
        else:
          # Process just the given line (testing purposes).
          df = self.Process_line(soupedEntry[givenLine], df)
    
      # Store df for later use
      self.df = df #FIXME: better name. How shall we store and exchange between classes?

  def Process_line(self, givenLine, df):
    """Processes a given XML pedecerto line. Puts syllable and length in a dataframe.

    Args:
        givenLine (xml): pedecerto xml encoding of a line of poetry
        df (dataframe): to store the data in

    Returns:
        dataframe: with syllables and their lenght (and some other information)
    """      
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
        newLine = {'line': current_line, 'syllable': current_syllable, 'length': length}
        df = df.append(newLine, ignore_index=True)

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



