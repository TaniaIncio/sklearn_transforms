from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')



class Imputer():

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primero copiamos el dataframe de datos de entrada 'X'
        data = X.copy()
        # replace null values according to the analysis of plots for each feature
        data['HOURS_DATASCIENCE'].replace(np.nan, data['HOURS_DATASCIENCE'].min(), inplace = True)
        data['HOURS_BACKEND'].replace(np.nan, data['HOURS_BACKEND'].median(), inplace = True)
        data['HOURS_FRONTEND'].replace(np.nan, data['HOURS_FRONTEND'].median(), inplace = True)
        data['NUM_COURSES_BEGINNER_DATASCIENCE'].replace(np.nan, data['NUM_COURSES_BEGINNER_DATASCIENCE'].mean(), inplace = True)
        data['NUM_COURSES_BEGINNER_BACKEND'].replace(np.nan, data['NUM_COURSES_BEGINNER_BACKEND'].median(), inplace = True)
        data['NUM_COURSES_BEGINNER_FRONTEND'].replace(np.nan, data['NUM_COURSES_BEGINNER_FRONTEND'].median(), inplace = True)
        data['NUM_COURSES_ADVANCED_DATASCIENCE'].replace(np.nan, data['NUM_COURSES_ADVANCED_DATASCIENCE'].median(), inplace = True)
        data['NUM_COURSES_ADVANCED_BACKEND'].replace(np.nan, data['NUM_COURSES_ADVANCED_BACKEND'].median(), inplace = True)
        data['NUM_COURSES_ADVANCED_FRONTEND'].replace(np.nan, data['NUM_COURSES_ADVANCED_FRONTEND'].median(), inplace = True)
        data['AVG_SCORE_DATASCIENCE'].replace(np.nan, data['AVG_SCORE_DATASCIENCE'].median(), inplace = True)
        data['AVG_SCORE_BACKEND'].replace(np.nan, data['AVG_SCORE_BACKEND'].median(), inplace = True)
        data['AVG_SCORE_FRONTEND'].replace(np.nan, data['AVG_SCORE_FRONTEND'].median(), inplace = True)
        return data


class StandardData():

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primero copiamos el dataframe de datos de entrada 'X'
        data = X.copy()
        # normalizamos datos, con desviacion estandar = 1
        data.iloc[:,:-1] = StandardScaler().fit_transform(data.iloc[:,:-1])
        return data