#       ___       ___           ___     
#      /\__\     /\  \         /\__\    
#     /:/  /    /::\  \       /::|  |   
#    /:/  /    /:/\ \  \     /:|:|  |   
#   /:/  /    _\:\~\ \  \   /:/|:|__|__ 
#  /:/__/    /\ \:\ \ \__\ /:/ |::::\__\
#  \:\  \    \:\ \:\ \/__/ \/__/~~/:/  /
#   \:\  \    \:\ \:\__\         /:/  / 
#    \:\  \    \:\/:/  /        /:/  /  
#     \:\__\    \::/  /        /:/  /   
#      \/__/     \/__/         \/__/    

# Latin Scansion Model
# Philippe Bors and Luuk Nolden
# Leiden University 2021

from bs4 import BeautifulSoup
# from cltk.stem.latin.syllabifier import Syllabifier
from cltk.prosody.latin.syllabifier import Syllabifier

import pandas as pd
import string
import os

import utilities as util
from bak.scansion_constants import ScansionConstants

import pedecerto.rhyme as pedecerto

class Pedecerto_parser:
  """This class parses the Pedecerto XML into a dataframe which can be used for
  training models.
  """  
  # Needed variables
  author = ''
  title = ''

  df = pd.DataFrame()
  
  # utilities = utilities.Utility()
  constants = ScansionConstants()
  
  def __init__(self, path, givenLine):
    # Create pandas dataframe
    column_names = ["author", "text", "line", "syllable", "length"]
    # column_names = ["author", "text", "line", "syllable", "foot", "feet_pos", "length", "word_boundary", "metrical_feature"]
    self.df = pd.DataFrame(columns = column_names) #FIXME: bad practise to work with self.df. Only update at the end.
    
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
        # Clean the lines (done by MQDQ, don't know what it does exactly)
        soupedEntry = util.clean(soupedEntry('line'))
        if givenLine == -1:
          # Do the entire folder
          for line in range(len(soupedEntry)):
          # for line in range(4):
            print('Progress on', self.author, self.title, ':', round(line / len(soupedEntry) * 100, 2), "%")
            # Process the entry. It will append the line to the df
            self.df = self.ProcessLine(soupedEntry[line], self.df)
        else:
          # Process just the given line (testing purposes).
          self.df = self.ProcessLine(soupedEntry[givenLine], self.df)
    
    # Now add features to the dataframe
    # self.df = self.AddFeature_Diphthong(self.df)
    # self.df = self.AddFeature_Speech(self.df)
  
  # Returns the dataframe appended
  def ProcessLine(self, givenLine, df):

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
        mySyllable = word_syllable_list[i].lower()
        
        # If we still have scansions available
        if number_of_scansions > 0:

          foot = split_scansion[i][0]
          feet_pos = split_scansion[i][1]

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

        number_of_scansions -= 1

        # Append to dataframe
        newLine = {'line': current_line, 'syllable': mySyllable, 'length': length}
        df = df.append(newLine, ignore_index=True)

    return df

    # exit(0)

    # all_syllables = syllabify_line(givenLine)
    # # Flatten list (hack)
    # all_syllables = [item for sublist in all_syllables for item in sublist]

    # words = givenLine.find_all('word')

    # line = givenLine['name']

    # length_list = []

    # for word in words:
      
    #   # word_syllabified = syllabify_line(word.string)

    #   # print(word_syllabified)


    #   # We now want to split every syllable to match its scansion.
    #   scansion = word['sy']
    #   current_word = word.string
    #   # print(item)
    #   n = 2
    #   # print('syllable', [item[i:i+n] for i in range(0, len(item), n)])
    #   split_scansion = [scansion[i:i+n] for i in range(0, len(scansion), n)] # per two characters

    #   print(split_scansion, len(split_scansion))
      
    #   for syllable in all_syllables:
    #     if syllable in current_word:
    #       print(syllable, current_word)
    #       exit(0)

    #   exit(0)

    #   length_list.extend(myScansions)

    # print(all_syllables, len(all_syllables))
    # print(length_list, len(length_list))

    # if len(all_syllables) == len(length_list):
    #   for i in range(len(all_syllables)):

    #     foot = length_list[i][0]
    #     feet_pos = length_list[i][1]
    #     mySyllable = all_syllables[i].lower()

    #     if feet_pos == 'A':
    #       length = 1
    #     elif feet_pos == 'T':
    #       length = 1
    #     elif feet_pos == 'b':
    #       length = 0
    #     elif feet_pos == 'c':
    #       length = 0
    #     elif feet_pos == '':
    #       length = -1        

    #     print(all_syllables[i], length_list[i], length)

    #     newLine = {'author': self.author, 'text': self.title, 'line': line, 'syllable': mySyllable, 'foot': foot, 'feet_pos': feet_pos, 
    #       'length': length}
    #     df = df.append(newLine, ignore_index=True)


    # else:
    #   raise ValueError("Length mismatch!")

        # Now, fill the dataframe: TODO: split length in foot and length

    # exit(0)
      # exit(0)

      # for i in range(len(mySyllables)):
      #   mySyllable = mySyllables[i]
      #   # To remove punctuation.
      #   mySyllable = mySyllable.translate(str.maketrans('', '', string.punctuation))


      #   try:
      #     myScansion = myScansions[i]
      #     foot = myScansion[0]
      #     feet_pos = myScansion[1]
        
      #   if feet_pos == 'A':
      #     length = 1
      #   elif feet_pos == 'T':
      #     length = 1
      #   elif feet_pos == 'b':
      #     length = 0
      #   elif feet_pos == 'c':
      #     length = 0
      #   elif feet_pos == '':
      #     length = -1
      #   else:
      #     print('Error occured determining feet_pos of syllable')


        
        # # newLine = {'author': self.author, 'text': self.title, 'line': line, 'syllable': mySyllable, 'foot': foot, 'feet_pos': feet_pos, 
        # #   'length': length, 'word_boundary': myWb, 'metrical_feature': myMf2}        
        

    # return df

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



