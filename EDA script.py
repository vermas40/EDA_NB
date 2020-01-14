"""
Add box plots and doc strings 
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression,LogisticRegression


def num_cat(df):
    num = df.select_dtypes(include=[np.number]).columns
    cat = list(set(df.columns) - set(num))
    
    datatypes = {
        'numeric' : num,
        'categorical' : cat
    }
    
    return (datatypes)

def uni_bi_numeric(df,data_types,lower_threshold = 0.8):
    print(df[data_types['numeric']].describe())
    df = df[data_types['numeric']]\
        .corr()\
        .unstack()\
        .to_frame()
    
    
    df = df \
        .loc[(df[0] > lower_threshold) & (df[0] < 1),]\
        .sort_values(by=0)\
        .drop_duplicates(subset = 0,keep = 'first')\
        .reset_index()

    return (df)    
    

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
        

    path = os.getcwd()
    plt.savefig(path + '/Plots/'+ type_of_graph + '/' + figure_name +'.png', bbox_inches='tight')
    plt.close()
    return(1)
    
def PCA_func(df,type_PCA):
    
    #centering data around 0 and using train data attributes to scale test data as well
    if type_PCA == 'basic':
        sc = StandardScaler()
        df = sc.fit_transform(df)
        
        pca = PCA()
        df = pca.fit_transform(df)
    elif type_PCA == 'sparse':
        pca = TruncatedSVD(n_components = 20,n_iter = 7,random_state = 42)
        df = pca.fit_transform(df)
    
    explained_variance = pd.DataFrame(pca.explained_variance_ratio_)
    explained_variance['comp_num'] = range(1,len(explained_variance)+1)
    explained_variance = explained_variance.rename(columns = {0:'var_explained'})
    explained_variance['cum_var'] = explained_variance.loc[:,'var_explained'].cumsum(axis = 0)
    print('Printing Variance Explained vs # of components')
    plot_graph(explained_variance,['comp_num','cum_var'],'line')
    
    
    
    if type_PCA == 'sparse':
        #if first component itself is explaining more than 90 % then just take the first component only
        if explained_variance.loc[0,'cum_var'] > 0.9:
            df = pd.DataFrame(df[:,0])
        else:
            df = pd.DataFrame(df[:,explained_variance.loc[explained_variance['cum_var'] <= 0.9,'comp_num'] - 1])
    elif type_PCA == 'basic':
        if explained_variance.loc[0,'cum_var'] > 0.9:
            df = df[:,0]
        else:
            df = df[[explained_variance.loc[explained_variance['cum_var'] <= 0.9,'comp_num'] - 1]]
        
    return(df)

def missing_value_treatment(df):
    """
    Performing imputation for categorical & continuous
    """
    
    if df.dtypes == 'O':
        #replacing all the nan with the most common level in categorical data
        df = df.fillna(df.value_counts().index[0])
    else:
        imputer = KNNImputer(n_neighbors=5)
        imputer.fit_transform(df)
    
    return (df)
    
def outlier_detection(df,data_types):
    """
    Local Outlier factor sees the distance of a particular point with it's immaediate neighbors.
    If the distance is very great then the point is classified as an outlier.
    """
    clf = LocalOutlierFactor(n_neighbors=20)
    clf.fit_predict(df[data_types['numeric']])
    X_scores = clf.negative_outlier_factor_
    df = df.iloc[~df.index.isin(np.where(np.absolute(np.around(X_scores,decimals = 0)) != 1)[0])].reset_index(drop=True)
    return(df)
    
def encoding(df):
    """
    In office post a question about why feature hasher works differently with different input types
    """
    le = LabelEncoder()
    df = df.apply(le.fit_transform)
    #feature_hasher = FeatureHasher(input_type = 'string')
    feature_hasher = FeatureHasher()
    hashed_df = feature_hasher.fit_transform(df.to_dict(orient='records'))
    return (hashed_df)

def feature_selection(df,target,exercise):
    X = df.iloc[:,~df.columns.isin([target])]
    y = df[[target]]
    
    if exercise == 'regression':
        estimator = LinearRegression()
    elif exercise == 'classification':
        estimator = LogisticRegression()
    
    selector = RFE(estimator,step=1)
    selector = selector.fit(X,y)
    
    pd.DataFrame(X.loc[:,selector.support_].columns,columns = ['imp_features']).to_csv(os.getcwd() + '\Feature_Selection\Imp_features.csv',index=False)
    return(X.loc[:,selector.support_].columns)

def EDA(name_of_dataset,target,type_exercise):
    
    os.chdir(os.path.dirname(os.path.realpath("EDA script")))
    dataset = pd.read_csv(name_of_dataset + '.csv')
    data_cache = num_cat(dataset)
    plot_dataset = uni_bi_numeric(dataset,data_cache,0.9)
    plot_dataset.loc[:,['level_0','level_1']].apply(lambda x_send: plot_graph(dataset,x_send,'scatter'),axis=1)
    print('Plotted scatter plots')
    dataset.loc[:,dataset.columns[dataset.isna().any()]] = dataset.loc[:,dataset.columns[dataset.isna().any()]].apply(missing_value_treatment)
    print('Performed missing value treatment')
    dataset.loc[:,data_cache['numeric']].apply(lambda x: plot_graph(x,['values','probability'],'before_outlier_density'),axis=0)
    print('Plotted PDFs')
    dataset_outlier_removed = outlier_detection(dataset,data_cache)
    print('Performed outlier treatment')
    dataset_outlier_removed.loc[:,data_cache['numeric']].apply(lambda x: plot_graph(x,['values','probability'],'after_outlier_density'),axis=0)
    print('Plotted PDFs after outlier treatment')
    dataset_outlier_removed_hashed_cat = encoding(dataset_outlier_removed.loc[:,set(data_cache['categorical']) - set(target)])
    pca_components_cat_data = PCA_func(dataset_outlier_removed_hashed_cat,'sparse')
    print('Performed PCA on categorical variables')
    dataset_categorical_pca = pd.concat([dataset_outlier_removed[data_cache['numeric']],pca_components_cat_data,dataset_outlier_removed[target]],axis=1)
    feature_selection(dataset_categorical_pca,target,type_exercise)
    print('Perfomed feature selection')
    
    return(1)
   
EDA("Crashes_Last_Five_Years","ALCOHOL_RELATED","classification")







