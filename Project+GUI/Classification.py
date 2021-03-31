"""
Classification:

1. Algorithms:
    * Logistic Regression
    * Random Forest
    * Support Vector Machine
    * Decision Tree

2. Evaluation:
    * Accuracy score
    * Precision
    * Recall
    * F1 score

3. Hold-out: 80% train, 20% test

4. k-fold cross-validation: k=5

"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.model_selection import cross_val_score
from joblib import dump, load

class Classification:

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def logistic_regression(self):
        log_reg = LogisticRegression(solver='liblinear', random_state=0)
        log_reg.fit(self.X_train, self.y_train)
        # save model
        dump(log_reg, 'LogisticRegression1.joblib')
        y_pred_test = log_reg.predict(self.X_test)
        y_pred_train = log_reg.predict(self.X_train)
        print('Logistic: Accuracy score on test set: {0:0.4f}'.format(accuracy_score(self.y_test, y_pred_test)))
        print('Logistic: Accuracy score on training set: {0:0.4f}'.format(accuracy_score(self.y_train, y_pred_train)))
        print('Logistic: Precision score: {0:0.4f}'.format(precision_score(self.y_test, y_pred_test)))
        print('Logistic: Recall score: {0:0.4f}'.format(recall_score(self.y_test, y_pred_test)))
        print('Logistic: F1 score: {0:0.4f}'.format(f1_score(self.y_test, y_pred_test)))
        # k-fold cross validation
        scores = cross_val_score(log_reg, self.X_train, self.y_train, cv=5, scoring='accuracy')
        print('Cross-validation scores: {}'.format(scores))

    def random_forest(self):
        clf_rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0)
        clf_rf.fit(self.X_train, self.y_train)
        # save model
        dump(clf_rf, 'RandomForest1.joblib')
        y_pred_test = clf_rf.predict(self.X_test)
        y_pred_train = clf_rf.predict(self.X_train)
        print('Random forest: Accuracy score on test set: {0:0.4f}'.format(accuracy_score(self.y_test, y_pred_test)))
        print('Random forest: Accuracy score on training set: {0:0.4f}'.format(accuracy_score(self.y_train, y_pred_train)))
        print('Random forest: Precision score: {0:0.4f}'.format(precision_score(self.y_test, y_pred_test)))
        print('Random forest: Recall score: {0:0.4f}'.format(recall_score(self.y_test, y_pred_test)))
        print('Random forest: F1 score: {0:0.4f}'.format(f1_score(self.y_test, y_pred_test)))
        # k-fold cross validation
        scores = cross_val_score(clf_rf, self.X_train, self.y_train, cv=5, scoring='accuracy')
        print('Cross-validation scores: {}'.format(scores))

    def support_vector_machine(self):
        clf_svm = svm.SVC()
        clf_svm.fit(self.X_train, self.y_train)
        # save model
        dump(clf_svm, 'SupportVectorMachine1.joblib')
        y_pred_test = clf_svm.predict(self.X_test)
        y_pred_train = clf_svm.predict(self.X_train)
        print('SVM: Accuracy score on test set: {0:0.4f}'.format(accuracy_score(self.y_test, y_pred_test)))
        print('SVM: Accuracy score on training set: {0:0.4f}'.format(accuracy_score(self.y_train, y_pred_train)))
        print('SVM: Precision score: {0:0.4f}'.format(precision_score(self.y_test, y_pred_test)))
        print('SVM: Recall score: {0:0.4f}'.format(recall_score(self.y_test, y_pred_test)))
        print('SVM: F1 score: {0:0.4f}'.format(f1_score(self.y_test, y_pred_test)))
        # k-fold cross validation
        scores = cross_val_score(clf_svm, self.X_train, self.y_train, cv=5, scoring='accuracy')
        print('Cross-validation scores: {}'.format(scores))

    def decisiontree(self):
        clf_dt = DecisionTreeClassifier(random_state=0)
        clf_dt.fit(self.X_train, self.y_train)
        # save model
        dump(clf_dt, 'DecisionTree1.joblib')
        y_pred_test = clf_dt.predict(self.X_test)
        y_pred_train = clf_dt.predict(self.X_train)
        print('Decision Tree: Accuracy score on test set: {0:0.4f}'.format(accuracy_score(self.y_test, y_pred_test)))
        print('Decision Tree: Accuracy score on training set: {0:0.4f}'.format(accuracy_score(self.y_train, y_pred_train)))
        print('Decision Tree: Precision score: {0:0.4f}'.format(precision_score(self.y_test, y_pred_test)))
        print('Decision Tree: Recall score: {0:0.4f}'.format(recall_score(self.y_test, y_pred_test)))
        print('Decision Tree: F1 score: {0:0.4f}'.format(f1_score(self.y_test, y_pred_test)))
        # k-fold cross validation
        scores = cross_val_score(clf_dt, self.X_train, self.y_train, cv=5, scoring='accuracy')
        print('Cross-validation scores: {}'.format(scores))