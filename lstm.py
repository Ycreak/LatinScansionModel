import os
import datetime
from progress.bar import Bar

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

import utilities as util

def convert_pedecerto_to_crf_array() -> list:

    if int(util.cf.get('Util', 'verbose')): print('Creating the sentence list')

    df = util.Pickle_read(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'pedecerto_df'))
    # Entire Aneid is put into a list. In this list, a list is dedicated per sentence.
    # Each sentence list has tuples consisting of a syllable and its length.

    # Convert the labels from int to str
    df['length'] = np.where(df['length'] == 0, 'short', df['length'])
    df['length'] = np.where(df['length'] == 1, 'long', df['length'])
    df['length'] = np.where(df['length'] == 2, 'elision', df['length'])

    all_sentences_list = []

    # Get number of books to process
    num_books = df['book'].max()

    # for i in range(num_books):
    for i in Bar('Processing').iter(range(num_books)):
        # Get only lines from this book
        current_book = i + 1
        book_df = df.loc[df['book'] == current_book]

        num_lines = book_df['line'].max()

        for j in range(num_lines):
            current_line = j + 1

            filtered_df = book_df[book_df["line"] == current_line]

            # length_list = filtered_df['length'].to_numpy()
            syllable_list = filtered_df['syllable'].to_numpy()
            # join them into 2d array and transpose it to get the correct crf format:
            # combined_list = np.array((syllable_list,length_list)).T
            # Append all to the list which we will return later
            all_sentences_list.append(syllable_list)

    return all_sentences_list

pedecerto_df = util.Pickle_read(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'pedecerto_df'))

# array = convert_pedecerto_to_crf_array()
# util.Pickle_write(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'test'), array)
array = util.Pickle_read(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'test'))

text_dataset = tf.data.Dataset.from_tensor_slices(["foo", "bar", "baz"])

vocab_data = [["luuk", "frank", "kaas", "pizza"],["earth", "wind", "and", "fire"]]

encoder = tf.keras.layers.TextVectorization(vocabulary=vocab_data)
# Now call an adapt for every item in the dataset
# encoder.adapt(text_dataset.batch(64))


exit(0)

class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])


mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)

df = pd.read_csv(csv_path)
# Slice [start:stop:step], starting from index 5 take every 6th record.
df = df[5::6]

date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')

print(df.head())

# plot_cols = ['T (degC)', 'p (mbar)', 'rho (g/m**3)']
# plot_features = df[plot_cols]
# plot_features.index = date_time
# _ = plot_features.plot(subplots=True)

# plot_features = df[plot_cols][:480]
# plot_features.index = date_time[:480]
# _ = plot_features.plot(subplots=True)


print(df.describe().transpose())

wv = df['wv (m/s)']
bad_wv = wv == -9999.0
wv[bad_wv] = 0.0

max_wv = df['max. wv (m/s)']
bad_max_wv = max_wv == -9999.0
max_wv[bad_max_wv] = 0.0

# The above inplace edits are reflected in the DataFrame.
df['wv (m/s)'].min()

wv = df.pop('wv (m/s)')
max_wv = df.pop('max. wv (m/s)')

# Convert to radians.
wd_rad = df.pop('wd (deg)')*np.pi / 180

# Calculate the wind x and y components.
df['Wx'] = wv*np.cos(wd_rad)
df['Wy'] = wv*np.sin(wd_rad)

# Calculate the max wind x and y components.
df['max Wx'] = max_wv*np.cos(wd_rad)
df['max Wy'] = max_wv*np.sin(wd_rad)

timestamp_s = date_time.map(pd.Timestamp.timestamp)

day = 24*60*60
year = (365.2425)*day

df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
plt.plot(np.array(df['Day sin'])[:25])
plt.plot(np.array(df['Day cos'])[:25])
plt.xlabel('Time [h]')
plt.title('Time of day signal')

# plt.show()

column_indices = {name: i for i, name in enumerate(df.columns)}

# Create train and test
n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]

train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
_ = ax.set_xticklabels(df.keys(), rotation=90)

plt.show()
