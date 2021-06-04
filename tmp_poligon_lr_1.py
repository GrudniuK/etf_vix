#https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_display_object_visualization.html#sphx-glr-auto-examples-miscellaneous-plot-display-object-visualization-py

import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = fetch_openml(data_id=1464, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

clf = make_pipeline(StandardScaler(), LogisticRegression(random_state=0))
clf.fit(X_train, y_train)



from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
              precision    recall  f1-score   support

           1       0.78      0.97      0.87       142
           2       0.64      0.16      0.25        45

    accuracy                           0.78       187
   macro avg       0.71      0.56      0.56       187
weighted avg       0.75      0.78      0.72       187


##########################3

clf = LogisticRegression(random_state=0)

X_train_new = pd.concat([X_train, y_train], axis = 1)
X_train_new.shape
X_train.shape

X_test_new = pd.concat([X_test, y_test], axis = 1)
X_test_new.shape
X_test.shape

clf.fit(X_train_new, y_train)

y_pred = clf.predict(X_test_new)
cm = confusion_matrix(y_test, y_pred)
cm
print(classification_report(y_test, y_pred))

clf.coef_


