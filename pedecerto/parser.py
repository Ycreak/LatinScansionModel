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
    column_names = ["author", "text", "line", "syllable", "foot", "feet_pos", "length"]
    # column_names = ["author", "text", "line", "syllable", "foot", "feet_pos", "length", "word_boundary", "metrical_feature"]
    self.df = pd.DataFrame(columns = column_names) #FIXME: bad practise to work with self.df. Only update at the end.
    
    # Add all entries to process to a list
    entries = self.CreateFilesList(path, 'xml')
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
          # for line in range(len(soupedEntry)):
          for line in range(2):
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
    syllabifier = Syllabifier()

    words = givenLine.find_all('word')

    line = givenLine['name']

    for word in words:
      
      myWord = word.string
      mySyllables = syllabifier.syllabify(myWord.lower())
      # We now want to split every syllable to match its scansion.
      item = word['sy']
      n = 2
      # print('syllable', [item[i:i+n] for i in range(0, len(item), n)])
      myScansions = [item[i:i+n] for i in range(0, len(item), n)]

      # try:
      #   # print('word boundary', word['wb'])
      #   myWb = word['wb']
      # except:
      #   # print("empty field") 
      #   myWb = ''
      # try:
      #   # print('metrical feature', word['mf'])
      #   myMf = word['mf']

      # except:
      #   # print("empty field")   
      #   myMf = ''

      # print('-------------------------------')

      for i in range(len(mySyllables)):
        mySyllable = mySyllables[i]
        # To remove punctuation.
        mySyllable = mySyllable.translate(str.maketrans('', '', string.punctuation))


        try:
          myScansion = myScansions[i]
          foot = myScansion[0]
          feet_pos = myScansion[1]
          # No metrical feature, so leave field empty
          # myMf2 = ''

        except:
          myScansion = ''
          foot = feet_pos = ''
          # Add the reason for this emptiness
          # myMf2  = myMf
        
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
        else:
          print('Error occured determining feet_pos of syllable')

        # Now, fill the dataframe: TODO: split length in foot and length
        newLine = {'author': self.author, 'text': self.title, 'line': line, 'syllable': mySyllable, 'foot': foot, 'feet_pos': feet_pos, 
          'length': length}
        
        # newLine = {'author': self.author, 'text': self.title, 'line': line, 'syllable': mySyllable, 'foot': foot, 'feet_pos': feet_pos, 
        #   'length': length, 'word_boundary': myWb, 'metrical_feature': myMf2}        
        
        df = df.append(newLine, ignore_index=True)

    return df

  def AddFeature_Speech(self, df):
    df['liquids'] = 0
    df['nasals'] = 0
    df['fricatives'] = 0
    # df['clusterable'] = 0
    df['mutes'] = 0
    df['aspirate'] = 0
    df['doubled_consonant'] = 0

    df['char_first'] = 0
    df['char_second'] = 0
    df['char_ultima'] = 0
    df['char_penultima'] = 0

    for i in range(len(df)):
      if any(liquid in df["syllable"][i] for liquid in self.constants.LIQUIDS):
        df['liquids'][i] = 1
      if any(nasal in df["syllable"][i] for nasal in self.constants.NASALS):
        df['nasals'][i] = 1
      if any(fricative in df["syllable"][i] for fricative in self.constants.FRICATIVES):
        df['fricatives'][i] = 1
      if any(mute in df["syllable"][i] for mute in self.constants.MUTES):
        df['mutes'][i] = 1        
      if any(aspirate in df["syllable"][i] for aspirate in self.constants.ASPIRATES):
        df['mutes'][i] = 1    
      
      df = self.CheckConsonantStatus(df["syllable"][i], df, i)

    return df

  def CheckConsonantStatus(self, string, df, i):
    
    char_first = string[0] 
    char_ultima = string[-1] 

    try:
      char_second = string[1]
      char_penultima = string[-2]
    except:
      char_second = '-'
      char_penultima = '-'
      print('String probably one character, continuing')

    if char_first in self.constants.CONSONANTS:
      print('first char is consonant')
      df['cons_first'][i] = 1

    if char_second in self.constants.CONSONANTS:
      print('second char is consonant')
      df['cons_second'][i] = 1

    if char_ultima in self.constants.CONSONANTS:
      print('last char is consonant')
      df['cons_ultima'][i] = 1

    if char_penultima in self.constants.CONSONANTS:
      print('second to last char is consonant')
      df['cons_penultima'][i] = 1

    return df


  def AddFeature_Diphthong(self, df):
    """Adds a diphtong feature to the given dataframe based on the syllable column

    Args:
        df (dataframe): does what it says on the tin

    Returns:
        df: dataframe with diphtong column appended
    """    
    # Initialise diphthong column to 0.
    df['diphtong'] = 0
    
    for i in range(len(df)):
      if any(diphtong in df["syllable"][i] for diphtong in self.constants.DIPTHONGS):
        df['diphtong'][i] = 1
        
    return df

  def CreateFilesList(self, path, extension):
    """Creates a list of files to be processed

    Args:
        path (string): folder to be searched
        extension (string): extension of files to be searched

    Returns:
        list: list with files to be searched
    """
    list = []
    
    for file in os.listdir(path):
        if file.endswith(".xml"):
            list.append(file)    
    
    return list

