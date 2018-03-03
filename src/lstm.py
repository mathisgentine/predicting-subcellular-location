from data_pipeline import get_sequences, N_CLASSES
from sklearn.model_selection import cross_val_score
import numpy as np
from keras.models import Model, Input
from keras.layers import LSTM, Dense, Dropout, Activation, Embedding
from keras.optimizers import RMSprop
from utils import get_val_split

BATCH_SIZE = 128
EPOCHS = 10


def get_model(input_dim, n_classes, embedding_dim=20):
    sequences = Input(shape=(None,))
    embedding = Embedding(input_dim=input_dim,
                          output_dim=embedding_dim)(sequences)  # Shape: (batch_size, seq_len, embedding_dim)
    lstm = LSTM(100,
                dropout=0,
                recurrent_dropout=0)(embedding)
    dense = Dense(n_classes)(lstm)
    dense = Activation('softmax')(dense)

    model = Model(inputs=sequences, outputs=dense)
    model.compile(optimizer=RMSprop(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


x_train, y_train, x_test, class_dict = get_sequences()
train_indices, val_indices = get_val_split(y_train)
x_val, y_val = x_train[val_indices, :], y_train[val_indices]
x_train, y_train = x_train[train_indices, :], y_train[train_indices]
print(x_train.shape)
input_dim = max(np.max(x_train), np.max(x_val))
print(input_dim)

model = get_model(input_dim+1, N_CLASSES)
model.fit(x=x_train,
          y=y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_data=(x_val, y_val))
cv_score = cross_val_score(model, x_train, y_train, cv=5, scoring='f1_micro')
print('CV score. Mean: {}. Sd: {}'.format(np.mean(cv_score), np.std(cv_score)))
