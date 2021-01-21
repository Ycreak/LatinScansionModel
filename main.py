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

import parser
import utilities

utilities = utilities.Utility()

# Provide the folder with Pedecerto XML files here. These will be put in a pandas dataframe
path = './texts'
# Optionally, provide a single line for testing purposes
line = -1
# Now call the parser and save the dataframe it creates
parse = parser.Parser(path, line)

df = parse.df
print(df)
exit(0)

# Now replace encoding by short and long
df['length'] = np.where(df['length'] == 'A', 1, df['length'])
df['length'] = np.where(df['length'] == 'T', 1, df['length'])
df['length'] = np.where(df['length'] == 'b', 0, df['length'])
df['length'] = np.where(df['length'] == 'c', 0, df['length'])

print(df)

# Model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing

df = df.drop(['author', 'text', 'line','word_boundary'], axis=1)

# Make estimator and estimee
X = df.drop('length', axis=1)
y = df['length']

# Make train and testset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf = RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))