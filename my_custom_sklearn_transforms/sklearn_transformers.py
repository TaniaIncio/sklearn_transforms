from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

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


class AddColumns(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        data['EQ_TOTAL'] = data['EQ_OFICINA'] + data['EQ_TRANSPORTE']
        return data


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
        data = StandardScaler().fit_transform(data)
        return data
        

class ImputerValues():

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primero copiamos el dataframe de datos de entrada 'X'
        df = X.copy()
        # replace null values according to the analysis of plots for each feature
        for cn in self.columns:
            df[cn].replace(np.nan, df[cn].median(), inplace = True)
        return df


class RemoveOutliersTotal():
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        self.y = y
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        df = X.copy()
        y = self.y.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        for cn in self.columns:
            v_sosp = df[cn].values
            q25, q75 = np.percentile(v_sosp, 25), np.percentile(v_sosp, 75)
            print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
            v_iqr = q75 - q25
            print('iqr: {}'.format(v_iqr))

            v_cut_off = v_iqr * 1.5
            v_lower, v_upper = q25 - v_cut_off, q75 + v_cut_off
            print('Cut Off: {}'.format(v_cut_off))
            print('v Lower: {}'.format(v_lower))
            print('v Upper: {}'.format(v_upper))

            outliers = [x for x in v_sosp if x < v_lower or x > v_upper]
            #print('Feature '+cn+' Outliers for Aceptado Cases: {}'.format(len(outliers)))
            #print('CXC outliers:{}'.format(outliers))

            df = df.drop(df[(df[cn] > v_upper) | (df[cn] < v_lower)].index)
        return df


class OutlierExtractor(TransformerMixin):
    def __init__(self, **kwargs):
        """
        Create a transformer to remove outliers. A threshold is set for selection
        criteria, and further arguments are passed to the LocalOutlierFactor class

        Keyword Args:
            neg_conf_val (float): The threshold for excluding samples with a lower
               negative outlier factor.

        Returns:
            object: to be used as a transformer method as part of Pipeline()
        """

        self.threshold = kwargs.pop('neg_conf_val', -10.0)

        self.kwargs = kwargs

    def transform(self, X, y):
        """
        Uses LocalOutlierFactor class to subselect data based on some threshold

        Returns:
            ndarray: subsampled data

        Notes:
            X should be of shape (n_samples, n_features)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        lcf = LocalOutlierFactor(**self.kwargs)
        lcf.fit(X)
        return (X[lcf.negative_outlier_factor_ > self.threshold, :],
                y[lcf.negative_outlier_factor_ > self.threshold])

    def fit(self, *args, **kwargs):
        return self


class RemoveOutliersPosClass():
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        self.y = y
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        df = X.copy()
        y = self.y.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        for cn in self.columns:
            v_sosp = df[cn].loc[y['OBJETIVO'] == 'Sospechoso'].values
            q25, q75 = np.percentile(v_sosp, 25), np.percentile(v_sosp, 75)
            print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
            v_iqr = q75 - q25
            print('iqr: {}'.format(v_iqr))

            v_cut_off = v_iqr * 1.5
            v_lower, v_upper = q25 - v_cut_off, q75 + v_cut_off
            print('Cut Off: {}'.format(v_cut_off))
            print('v Lower: {}'.format(v_lower))
            print('v Upper: {}'.format(v_upper))

            outliers = [x for x in v_sosp if x < v_lower or x > v_upper]
            #print('Feature '+cn+' Outliers for Aceptado Cases: {}'.format(len(outliers)))
            #print('CXC outliers:{}'.format(outliers))

            df = df.drop(df[(df[cn] > v_upper) | (df[cn] < v_lower)].index)
        return df
    

class RemoveOutliersNegClass():
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        self.y = y
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        df = X.copy()
        y = self.y.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        for cn in self.columns:
            v_sosp = df[cn].loc[y['OBJETIVO'] == 'Aceptado'].values
            q25, q75 = np.percentile(v_sosp, 25), np.percentile(v_sosp, 75)
            print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
            v_iqr = q75 - q25
            print('iqr: {}'.format(v_iqr))

            v_cut_off = v_iqr * 1.5
            v_lower, v_upper = q25 - v_cut_off, q75 + v_cut_off
            print('Cut Off: {}'.format(v_cut_off))
            print('v Lower: {}'.format(v_lower))
            print('v Upper: {}'.format(v_upper))

            outliers = [x for x in v_sosp if x < v_lower or x > v_upper]
            #print('Feature '+cn+' Outliers for Aceptado Cases: {}'.format(len(outliers)))
            #print('CXC outliers:{}'.format(outliers))

            df = df.drop(df[(df[cn] > v_upper) | (df[cn] < v_lower)].index)
        return df
