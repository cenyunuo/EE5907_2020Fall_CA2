import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def CNN(X_train, y_train, X_test, y_test, epochs):
    model = keras.Sequential([layers.Conv2D(20, 5, strides=(1, 1), padding ='same', activation='relu',
                                            kernel_regularizer=keras.regularizers.l2(0.001),
                                            kernel_initializer=keras.initializers.GlorotNormal(),
                                            input_shape=(32, 32, 1)),
                              layers.MaxPooling2D(pool_size=(2, 2)),
                              layers.Conv2D(50, 5, strides=(1, 1), padding ='same', activation='relu',
                                            kernel_regularizer=keras.regularizers.l2(0.001),
                                            kernel_initializer=keras.initializers.GlorotNormal()),
                              layers.MaxPooling2D(pool_size=(2, 2), name='pooling_2'),
                              layers.Flatten(),
                              layers.Dense(500, activation = 'relu', name = 'dense_1'),
                              layers.Dropout(0.5),
                              layers.Dense(26, activation = 'softmax', name='dense_2')])

    model.compile(optimizer=keras.optimizers.RMSprop(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    #Normalized the value to (0,1)
    X_train = X_train.reshape(len(y_train),32,32,1).astype('float32')/255
    X_test = X_test.reshape(len(y_test),32,32,1).astype('float32')/255
    #Move the label to {0, 1, ..., 25}
    y_train = y_train - 1
    y_test = y_test - 1
    #Fit the model
    history = model.fit(X_train, y_train, batch_size=42, epochs=epochs, validation_data=(X_test, y_test))
    test_scores = model.evaluate(X_test, y_test, verbose=2)
    print("Test loss: {}; Test accuracy: {}", test_scores[0], test_scores[1])
    return model, history
