from sklearn.ensemble import RandomForestClassifier
from data_pipeline import get_handcrafted_data
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from utils import get_test_split, get_val_split, plot_confusion_matrix, plot_roc_curve
from sklearn.metrics import f1_score

DO_CV = True

x, y, x_blind, class_dict = get_handcrafted_data()
train_idxs, test_idxs = get_test_split(y)
x_train, y_train = x[train_idxs, :], y[train_idxs]
x_test, y_test = x[test_idxs, :], y[test_idxs]

if DO_CV:
    model = RandomForestClassifier(n_estimators=150, min_samples_leaf=5)  # , probability=True
    cv_score = cross_val_score(model, x_train, y_train, cv=4, scoring='accuracy')  # f1_micro
    print('CV score. Mean: {}. Sd: {}'.format(np.mean(cv_score), np.std(cv_score)))
else:
    model = RandomForestClassifier(random_state=0)
    model.fit(x_train, y_train)
    test_score = model.score(x_test, y_test)
    print('Test score: {}'.format(test_score))

    class_dict_inv = {v: k for k, v in class_dict.items()}
    y_pred = model.predict(x_test)
    f1s = f1_score(y_test, y_pred, average=None)
    print('F1 scores:')
    for k, i in class_dict.items():
        print('{}: {}'.format(k, f1s[i]))
    plot_confusion_matrix(y_pred, y_test, [class_dict_inv[i] for i in range(len(class_dict))], normalize=True)
    y_pred = model.predict_proba(x_test)
    plot_roc_curve(y_pred, y_test, class_dict)
