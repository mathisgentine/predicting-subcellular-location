from data_pipeline import get_handcrafted_data, N_CLASSES
from sklearn.model_selection import cross_val_score
import numpy as np
from keras.models import Model, Input
from keras.layers import LSTM, Dense, Dropout, Activation, Embedding
from keras.optimizers import Adam
from utils import get_val_split
from keras.models import load_model


BATCH_SIZE = 128
EPOCHS = 70


def get_model(input_dim, n_classes):
    nn_input = Input(shape=(input_dim,))
    dense = Dense(400, activation='relu')(nn_input)
    dense = Dropout(0.5)(dense)
    dense = Dense(200, activation='relu')(dense)
    dense = Dropout(0.5)(dense)
    dense = Dense(100, activation='relu')(dense)
    dense = Dropout(0.5)(dense)
    dense = Dense(50, activation='relu')(dense)
    dense = Dropout(0.5)(dense)
    dense = Dense(n_classes)(dense)
    dense = Activation('softmax')(dense)

    model = Model(inputs=nn_input, outputs=dense)
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


x_train, y_train, x_test, class_dict = get_handcrafted_data(one_hot=True)
train_indices, val_indices = get_val_split(y_train)
x_val, y_val = x_train[val_indices, :], y_train[val_indices]
x_train, y_train = x_train[train_indices, :], y_train[train_indices]
print(x_train.shape)

input_dim = x_train.shape[1]

model = get_model(input_dim, N_CLASSES)
model.fit(x=x_train,
          y=y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_data=(x_val, y_val))

cv_score = model.evaluate(x_val, y_val)[1]
print('CV score: {}'.format(cv_score))

model.save('../checkpoints/model.h5')
del model
model = load_model('../checkpoints/model.h5')
