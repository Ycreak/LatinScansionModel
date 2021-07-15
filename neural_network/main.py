from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

LOAD_PATH = ''
HIDDEN_SIZE = 256

def load_data(load_path):
    # Todo :)
    return

def create_model(X_train, y_train):
    """Create a neural network with two hidden layers,
         Dependent on the sizes of the training data

    Args:
        X_train (array): .
        y_train (array): .

    Returns:
        ...
    """    
    model = Sequential()
    model.add(Dense(HIDDEN_SIZE, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(HIDDEN_SIZE, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model

def main():
    # Load from file or re-generate
    X, y = load_data(load_path=LOAD_PATH)
    print('-- Data size --\nX: {} \ny: {}'.format(X.shape, y.shape))

    # Split test and train set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Scale data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create the model
    model = create_model(X_train, y_train)

    # # Train
    history = model.fit(X_train, y_train, epochs=80, batch_size=10, validation_data=(X_test, y_test), shuffle=True)


    _, train_accuracy = model.evaluate(X_train, y_train)
    _, test_accuracy = model.evaluate(X_test, y_test)

    print('Accuracy (training): %.2f' % (train_accuracy * 100))
    print('Accuracy (testing): %.2f' % (test_accuracy * 100))

    # # plot loss during training
    # pyplot.subplot(211)
    # pyplot.title('Loss')
    # pyplot.xlabel('epoch')
    # pyplot.plot(history.history['loss'], label='train')
    # pyplot.plot(history.history['val_loss'], label='test')
    # pyplot.legend()

    # # plot accuracy during training
    # pyplot.subplot(212)
    # pyplot.title('Accuracy')
    # pyplot.xlabel('epoch')
    # pyplot.plot(history.history['binary_accuracy'], label='train')
    # pyplot.plot(history.history['val_binary_accuracy'], label='test')
    # pyplot.legend()

    # pyplot.tight_layout()
    # pyplot.show()

    #model = create_model(X_train, y_train)


    # tuner = kt.Hyperband(model,
    #                  objective='val_binary_accuracy',
    #                  max_epochs=10,
    #                  factor=3,
    #                  directory='./',
    #                  project_name='intro_to_kt')


if __name__ == "__main__":
    main()