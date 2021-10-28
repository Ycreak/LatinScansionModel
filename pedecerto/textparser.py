from bs4 import BeautifulSoup

import pandas as pd
import string
import os
import timeit
from progress.bar import Bar
import copy
import utilities as util

import pedecerto.rhyme as pedecerto

#NB: line 1069 from Rerum Natura deleted

class Pedecerto_parser:
  """This class parses the Pedecerto XML into a dataframe which can be used for
  training models.

  NB: the XML files are stripped of their headers, leaving the body to be processed
  """  
  def __init__(self, path):
 
    # Create pandas dataframe
    column_names = ["title", "line", "syllable", "length"]
    df = pd.DataFrame(columns = column_names) 
    
    # Add all entries to process to a list
    entries = util.Create_files_list(path, 'xml')
    # Process all entries added to the list
    for entry in entries:
      with open(path + entry) as fh:
        # for each text, an individual dataframe will be created and saved as pickle
        new_text_df = copy.deepcopy(df)
        pickle_name = 'syllable_label_' + entry + '.pickle'

        # Use beautiful soup to process the xml
        soupedEntry = BeautifulSoup(fh,"xml")
        # Retrieve the title and author from the xml file
        text_title = str(soupedEntry.title.string)
        author = str(soupedEntry.author.string)
        # Clean the lines (done by MQDQ)
        soupedEntry = util.clean(soupedEntry('line'))

        # for line in range(len(soupedEntry)):
        for line in Bar('Processing {0}, {1}'.format(author, text_title)).iter(range(len(soupedEntry))):
          book_title = int(soupedEntry[line].parent.get('title'))
          # Process the entry. It will append the line to the df
          if not soupedEntry[line]['name'].isdigit():
            continue # If our line name is not a digit, the line is uncertain. We skip over it  
          line_df = self.Process_line(soupedEntry[line], book_title)
          new_text_df = new_text_df.append(line_df, ignore_index=True) # If I greatly improve my own code, am I a wizard, or a moron?
        
        # Clean the lines that did not succeed
        new_text_df = self.clean_generated_df(new_text_df)

        util.Pickle_write(util.cf.get('Pickle', 'path'), pickle_name, new_text_df)

  def clean_generated_df(self, df):
    # Processes all lines in the given df and deletes the line if there is an ERROR reported
    for title_index in Bar('Cleaning dataframe').iter(range(df['title'].max())):
        # Get only lines from this book
        title_df = df.loc[df['title'] == title_index + 1]
        # Per book, process the lines
        for line_index in range(title_df['line'].max()):
            line_df = title_df[title_df["line"] == line_index + 1]
            if 'ERROR' in line_df['syllable'].values:
                # Now delete this little dataframe from the main dataframe
                keys = list(line_df.columns.values)
                i1 = df.set_index(keys).index
                i2 = line_df.set_index(keys).index
                df = df[~i1.isin(i2)]

    return df


  def Process_line(self, given_line, book_title):
    """Processes a given XML pedecerto line. Puts syllable and length in a dataframe.

    Args:
        given_line (xml): pedecerto xml encoding of a line of poetry
        df (dataframe): to store the data in
        book_title (str): title of the current book (Book 1)

    Returns:
        dataframe: with syllables and their lenght (and some other information)
    """      
    column_names = ["title", "line", "syllable", "length"]
    df = pd.DataFrame(columns = column_names)

    current_line = given_line['name']

    # Parse every word and add its features
    for w in given_line("word"):
      
      # Now for every word, syllabify it first
      try:
        word_syllable_list = pedecerto._syllabify_word(w)
      except:
        new_line = {'title': int(book_title), 'line': int(current_line), 'syllable': 'ERROR', 'length': -1}
        df = df.append(new_line, ignore_index=True)
        return df

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

          feet_pos = split_scansion[i][1]

          # Interpret length based on pedecerto encoding (could be done much quicker)
          if feet_pos.isupper():
            length = 1
          elif feet_pos.islower():
            length = 0
        # No scansions available? Elision. Denote with 2
        else:
          length = 2

        # Keep track of performed operations
        number_of_scansions -= 1

        # Append to dataframe
        new_line = {'title': int(book_title), 'line': int(current_line), 'syllable': current_syllable, 'length': int(length)}
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



