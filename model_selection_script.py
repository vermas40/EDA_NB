"""
This notebook is just for implement multiclass models out of the box without any feature selection
& feature engineering

We will try KNN, Naive Bayes, Decision Trees, Random Forest, XGBoost, Neural Nets
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def plot_graph(main_dataset,axis_label,type_of_graph):
    """
    Earlier call was plot_graph(main_dataset,figure_obj,x,type_of_graph,plotting_dataset = [1])
    """
    if type_of_graph == 'scatter':
        #ax = figure_obj.add_subplot(len(plotting_dataset),1,x[2])
        plt.scatter(main_dataset[axis_label[0]],main_dataset[axis_label[1]])
        plt.xlabel(axis_label[0])
        plt.ylabel(axis_label[1])
        figure_name = axis_label[0]+' vs '+axis_label[1]
    elif type_of_graph == 'line':
        #ax = figure_obj.add_subplot(len(plotting_dataset),1,1)
        plt.plot(main_dataset[axis_label[0]],main_dataset[axis_label[1]])
        plt.xlabel(axis_label[0])
        plt.ylabel(axis_label[1])
        figure_name = axis_label[0]+' vs '+axis_label[1]
    elif type_of_graph == 'before_outlier_density':
        figure_name = main_dataset.name
        #main_dataset.plot(kind='density')
        density = gaussian_kde(main_dataset)
        output_axis = np.linspace(min(main_dataset),max(main_dataset),100)
        plt.plot(output_axis,density(output_axis))
        plt.xlabel(axis_label[0])
        plt.ylabel(axis_label[1]) 

        
    elif type_of_graph == 'after_outlier_density':
        density = gaussian_kde(main_dataset)
        output_axis = np.linspace(min(main_dataset),max(main_dataset),100)
        plt.plot(output_axis,density(output_axis))
        plt.xlabel(axis_label[0])
        plt.ylabel(axis_label[1])        
        figure_name = main_dataset.name
    
    elif type_of_graph == 'bar chart':
        plt.bar(x=[i for i in range(0,len(main_dataset))],\
                   height=main_dataset[main_dataset.select_dtypes(include=[np.number]).columns[0]],\
                   tick_label=main_dataset[main_dataset.select_dtypes(include=[object]).columns[0]])

        plt.xlabel(axis_label[0])
        plt.xlabel(axis_label[1])
        figure_name = axis_label[0]+' vs '+axis_label[1]

    path = os.getcwd()
    plt.savefig(path + '/Plots/'+ type_of_graph + '/' + figure_name +'.png', bbox_inches='tight')
    plt.close()
    return(1)
 
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
        
def model_selection(name_of_dataset,target_name):
    os.chdir(r'C:\Users\shivam.verma\Documents\Side Hoes\Model_Selection_NB\Machine_Learning')
    dataset = pd.read_csv(name_of_dataset + '.csv')
    X,y = x_y_split(dataset,target_name)
    X_train_set,X_test_set,y_train_set,y_test_set = pre_process(X,y)
    classifier_list = ['knn','naive','DT','RF','NN']
    plot_df=classify_compare(X_train_set,X_test_set,y_train_set,y_test_set,classifier_list,42)      
    plot_graph(plot_df,['Classifier','Score',],'bar chart')     
    return(1)
    
model_selection('Crashes_Last_Five_Years','ALCOHOL_RELATED')


    
    
