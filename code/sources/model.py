import tensorflow as tf
from tensorflow import keras


def get_model(parameters):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(parameters['dense_units'], activation=parameters['activation']),
        keras.layers.Dropout(parameters['dropout']),
        keras.layers.Dense(parameters['dense_units'], activation=parameters['activation']),
        keras.layers.Dropout(parameters['dropout']),
        keras.layers.Dense(parameters['dense_units'], activation=parameters['activation']),
        keras.layers.Dropout(parameters['dropout']),
        keras.layers.Dense(10, activation='softmax')
    ])

    if parameters['optimizer'] == 'Adam':
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=parameters['learning_rate'],
        )
    elif parameters['optimizer'] == 'Nadam':
        optimizer = tf.keras.optimizers.Nadam(
            learning_rate=parameters['learning_rate'],
        )
    elif parameters['optimizer'] == 'SGD':
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=parameters['learning_rate'],
        )
    else:
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=parameters['learning_rate'],
        )

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
