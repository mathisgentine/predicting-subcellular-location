from sklearn.svm import SVC
from data_pipeline import get_handcrafted_data
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

x_train, y_train, x_test, class_dict = get_handcrafted_data()
print(x_train.shape)
model = SVC(C=50, probability=True)
cv_score = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')  # f1_micro
print('CV score. Mean: {}. Sd: {}'.format(np.mean(cv_score), np.std(cv_score)))
