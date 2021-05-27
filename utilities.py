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

import pickle

def Pickle_write(path, file_name, object):
    destination = path + file_name

    with open(destination, 'wb') as f:
        pickle.dump(object, f)

def Pickle_read(path, file_name):
    destination = path + file_name

    with open(destination, 'rb') as f:
        return pickle.load(f)

    # class Utility:
    # """This class provides utilities for the other classes.
    # """  
    # def __init__(self):
    #     print('Utilities called')

    # def clean(self, ll):

    #     """Remove all corrupt lines from a set of bs4 <line>s

    #     Args:
    #         ll (list of bs4 <line>): Lines to clean

    #     Returns:
    #         (list of bs4 <line>): The lines, with the corrupt ones removed.
    #     """

    #     return [
    #         l
    #         for l in ll
    #         if l.has_attr("pattern")
    #         and l["pattern"] != "corrupt"
    #         and l["pattern"] != "not scanned"
    #     ]