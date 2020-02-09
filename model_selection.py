"""
This notebook is just for implement multiclass models out of the box without any feature selection
& feature engineering

We will try KNN, Naive Bayes, Decision Trees, Random Forest, XGBoost, Neural Nets
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

os.chdir(r'C:\Users\shivam.verma\Documents\Side Hoes\Model_Selection')
iris = pd.read_csv('iris.csv')

def x_y_split(df,label):
    return df.loc[:,df.columns!=label],df.loc[:,[label]]

def pre_process(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train,X_test,y_train,y_test

def classify_compare(X_train,X_test,y_train,y_test,classifier_list,seed=42):
    df = pd.DataFrame(columns=['classifier','score'])

    for classifier in classifier_list:
        if classifier == 'knn':
            clf = KNeighborsClassifier(n_neighbors=10)
        elif classifier == 'naive':
            clf = GaussianNB()
        elif classifier == 'DT':
            clf = DecisionTreeClassifier(random_state = seed)
        elif classifier == 'RF':
            clf = RandomForestClassifier(random_state = seed)
        else:
            clf = MLPClassifier(random_state=seed)
        clf.fit(X_train,y_train)
        score_val = clf.score(X_test,y_test)
        df.loc[len(df)]=[classifier,score_val]
    return df

X,y = x_y_split(iris,"flower_class")

X_train_set,X_test_set,y_train_set,y_test_set = pre_process(X,y)
classifier_list = ['knn','naive','DT','RF','NN']
plot_df=classify_compare(X_train_set,X_test_set,y_train_set,y_test_set,classifier_list,42)      
    
        
        
        

    
    
