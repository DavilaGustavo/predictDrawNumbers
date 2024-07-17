from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import to_categorical

# Carregar e pré-processar os dados
(X_train, y_train), (X_test, y_test) = mnist.load_data()

previsores_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
previsores_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

previsores_train = previsores_train.astype('float32')
previsores_test = previsores_test.astype('float32')
previsores_train /= 255
previsores_test /= 255

classe_train = to_categorical(y_train, 10)
classe_test = to_categorical(y_test, 10)

# Criar a rede neural
classificador = Sequential()
classificador.add(Conv2D(32, (3,3), input_shape=(28, 28, 1), activation='relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))

classificador.add(Conv2D(32, (3,3), activation='relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))

classificador.add(Flatten())
classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=10, activation='softmax'))
classificador.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treinar o modelo
# Modelo utilizado no projeto possui 50 épocas
classificador.fit(previsores_train, classe_train, batch_size=128, epochs=5, validation_data=(previsores_test, classe_test))

# Salvar o modelo treinado
classificador_json = classificador.to_json()
with open('classificadorNumbers.json', 'w') as json_file:
    json_file.write(classificador_json)
classificador.save_weights('classificadorNumbers.weights.h5')