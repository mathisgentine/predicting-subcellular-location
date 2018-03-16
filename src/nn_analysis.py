from keras.models import load_model

model = load_model('../checkpoints/model.h5')
x_train, y_train, x_test, class_dict = get_handcrafted_data(one_hot=True)
train_indices, val_indices = get_val_split(y_train)
x_val, y_val = x_train[val_indices, :], y_train[val_indices]
x_train, y_train = x_train[train_indices, :], y_train[train_indices]

cv_score = model.evaluate(x_val, y_val)[1]
print('CV score: {}'.format(cv_score))
