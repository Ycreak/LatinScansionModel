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
from cltk.stem.latin.syllabifier import Syllabifier
import pandas as pd
import numpy as np
import string
import os
import utilities

class Parser:
  """This class parses the Pedecerto XML into a dataframe which can be used for
  training models.
  """  
  # Needed variables
  author = ''
  title = ''

  df = pd.DataFrame()
  
  utilities = utilities.Utility()

  
  def __init__(self, path, givenLine = -1):
    # Create pandas dataframe
    column_names = ["author", "text", "line", "syllable", "foot", "length", "word_boundary", "metrical_feature"]
    self.df = pd.DataFrame(columns = column_names)

    entries = []
    for file in os.listdir(path):
        if file.endswith(".xml"):
            entries.append(file)

    for entry in entries:
      with open('./texts/' + entry) as fh:
        soupedEntry = BeautifulSoup(fh,"xml")

        # Retrieve the title and author from the xml file
        self.title = soupedEntry.title.string
        self.author = soupedEntry.author.string

        # Clean the lines
        soupedEntry = self.utilities.clean(soupedEntry('line'))

        if givenLine == -1:
          # Do the entire folder
          for line in range(len(soupedEntry)):
            print('Progress on', self.author, self.title, ':', round(line / len(soupedEntry) * 100, 2), "%")
            # Process the entry. It will append the line to the df
            self.df = self.ProcessLine(soupedEntry[line], self.df)
        else:
          # Process just the given line.
          self.df = self.ProcessLine(soupedEntry[givenLine], self.df)
  
  # Returns the dataframe appended
  def ProcessLine(self, givenLine, df):
    syllabifier = Syllabifier()


    words = givenLine.find_all('word')

    line = givenLine['name']
    # print(line)
    # exit(0)

    for word in words:

      # print('word', word.string, syllabifier.syllabify(word.string))
      
      myWord = word.string
      mySyllables = syllabifier.syllabify(myWord)
      # We now want to split every syllable to match its scansion.
      item = word['sy']
      n = 2
      # print('syllable', [item[i:i+n] for i in range(0, len(item), n)])

      myScansions = [item[i:i+n] for i in range(0, len(item), n)]

      try:
        # print('word boundary', word['wb'])
        myWb = word['wb']
      except:
        # print("empty field") 
        myWb = ''
      try:
        # print('metrical feature', word['mf'])
        myMf = word['mf']

      except:
        # print("empty field")   
        myMf = ''

      # print('-------------------------------')

      for i in range(len(mySyllables)):
        mySyllable = mySyllables[i]
        # To remove punctuation.
        mySyllable = mySyllable.translate(str.maketrans('', '', string.punctuation))


        try:
          myScansion = myScansions[i]
          foot = myScansion[0]
          length = myScansion[1]
          # No metrical feature, so leave field empty
          myMf2 = ''

        except:
          myScansion = ''
          foot = length = ''
          # Add the reason for this emptiness
          myMf2 = myMf
        
        # Now, fill the dataframe: TODO: split length in foot and length
        newLine = {'author': self.author, 'text': self.title, 'line': line, 'syllable': mySyllable, 'foot': foot, 'length': length, 
          'word_boundary': myWb, 'metrical_feature': myMf2}
        df = df.append(newLine, ignore_index=True)

    return df





