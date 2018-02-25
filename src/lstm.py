from sklearn.svm import SVC
from data_pipeline import get_sequences, N_CLASSES
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from keras.models import Model, Input
from keras.layers import LSTM, Dense, Dropout, Activation
from keras.optimizers import RMSprop
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold


BATCH_SIZE = 128
EPOCHS = 10
N_FOLDS = 5


def get_model(input_dim, n_classes):
    sequences = Input(shape=(None, input_dim))
    lstm = LSTM(100,
                dropout=0,
                recurrent_dropout=0)(sequences)
    dense = Dense(n_classes)(lstm)
    dense = Activation('softmax')(dense)

    model = Model(inputs=sequences, outputs=dense)
    model.compile(optimizer=RMSprop(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


x_train, y_train, x_test, class_dict = get_sequences()
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True)
train_indices, val_indices = next(skf.split(x_train, y_train))
x_val, y_val = x_train[val_indices, :, :], y_train[val_indices]
x_train, y_train = x_train[train_indices, :, :], y_train[train_indices]

model = get_model(x_train[0].shape[1], N_CLASSES)
model.fit(x=x_train,
          y=y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_data=(x_val, y_val))
cv_score = cross_val_score(model, x_train, y_train, cv=5, scoring='f1_micro')
print('CV score. Mean: {}. Sd: {}'.format(np.mean(cv_score), np.std(cv_score)))
