import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.feature_extraction import FeatureHasher

# Graphics in retina format are more sharp and legible
#%config InlineBackend.figure_format = 'retina' 

# Increase the default plot size and set the color scheme
plt.rcParams['figure.figsize'] = 8, 5
plt.rcParams['image.cmap'] = 'viridis'

os.chdir('E:\\Libraries\\Documents\\Side Hoes\\EDA\\EDA\\EDA_NB')

dataset = pd.read_csv('Crashes_Last_Five_Years.csv')

def num_cat(df):
    num = df.select_dtypes(include=[np.number]).columns
    cat = list(set(df.columns) - set(num))
    
    datatypes = {
        'numeric' : num,
        'categorical' : cat
    }
    
    return datatypes

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
    

def plot_graph(main_dataset,x,type_of_graph):
    """
    Earlier call was plot_graph(main_dataset,figure_obj,x,type_of_graph,plotting_dataset = [1])
    """
    if type_of_graph == 'scatter':
        #ax = figure_obj.add_subplot(len(plotting_dataset),1,x[2])
        plt.scatter(main_dataset[x[0]],main_dataset[x[1]])
    elif type_of_graph == 'line':
        #ax = figure_obj.add_subplot(len(plotting_dataset),1,1)
        plt.plot(main_dataset[x[0]],main_dataset[x[1]])
    plt.xlabel(x[0])
    plt.ylabel(x[1])
    path = os.getcwd()
    plt.savefig(path + '/Plots/'+ type_of_graph + '/' + x[0]+' vs '+x[1]+'.png', bbox_inches='tight')
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
        pca.fit_transform(df)
    
    explained_variance = pd.DataFrame(pca.explained_variance_ratio_)
    explained_variance['comp_num'] = range(1,len(explained_variance)+1)
    explained_variance = explained_variance.rename(columns = {0:'var_explained'})
    explained_variance['cum_var'] = explained_variance.loc[:,'var_explained'].cumsum(axis = 0)
    print('Printing Variance Explained vs # of components')
    plot_graph(explained_variance,['comp_num','cum_var'],'line')
    
    
    
    if type_PCA == 'sparse':
        df = df\
        .toarray()\
        [:,explained_variance.loc[explained_variance['cum_var'] <= 0.9,'comp_num'] - 1]
    elif type_PCA == 'basic':
        df = df[[explained_variance.loc[explained_variance['cum_var'] <= 0.9,'comp_num'] - 1]]
        
    return(df)

def missing_value_treatment(df):
    imputer = KNNImputer(n_neighbors=5)
    
    
def outlier_detection(out_ds,data_types):
    """
    Local Outlier factor sees the distance of a particular point with it's immaediate neighbors.
    If the distance is very great then the point is classified as an outlier.
    """
    out_ds = out_ds.fillna(0)    
    clf = LocalOutlierFactor(n_neighbors=20)
    clf.fit_predict(out_ds[data_types['numeric']])
    X_scores = clf.negative_outlier_factor_
    out_ds = out_ds.iloc[~out_ds.index.isin(np.where(np.absolute(np.around(X_scores,decimals = 0)) != 1)[0])]
    return(out_ds)
    
def encoding(df):
    le = LabelEncoder()
    df = df.apply(le.fit_transform)
    #feature_hasher = FeatureHasher(input_type = 'string')
    feature_hasher = FeatureHasher()
    hashed_df = feature_hasher.fit_transform(df)
    return (hashed_df)

target = 'ALCOHOL_RELATED'
data_cache = num_cat(dataset)
plot_dataset = uni_bi_numeric(dataset,data_cache,0.9)
plot_dataset.loc[:,['level_0','level_1']].apply(lambda x_send: plot_graph(dataset,x_send,'scatter'),axis=1)

dataset_outlier_removed = outlier_detection(dataset,data_cache)
dataset_outlier_removed_hashed_cat = encoding(dataset_outlier_removed.loc[:,set(data_cache['categorical']) - set(target)])
pca_components = PCA_func(dataset_outlier_removed_hashed_cat,'sparse')
