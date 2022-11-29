import numpy as np
import math
import tqdm
import xgboost as xgb
from skopt import BayesSearchCV
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, mean_absolute_error, make_scorer

class My_XGBImputer:

    def __init__(self, missing_values=np.nan, parameters={'learning_rate':0.1}):

        '''define class attributes'''
        self.missing_values = missing_values
        self.parameters = parameters
        self.best_parameters = []
        self.best_scores = []
        self.order = None
        self.models = []
        self.nan_cols = []

    def fit_transform(self, dataframe, scheme='default', order='ascending', nr_iterations = 100,
                        params=None, use_imputed=True, threshold=1.0, cv=3, split=0.3, RState=42):
        
        X = dataframe.columns[dataframe.isna().any() == False]
        self.nan_cols = dataframe[dataframe.isna().any()].columns.to_list()

        # remove columns when percentage of missing values is greater than the threshold
        for col in self.nan_cols:
            if dataframe[dataframe[col].isna().any()].shape[0]/dataframe.shape[0] > threshold:
                self.nan_cols.remolve(col)

        nan_col_data = dataframe[self.nan_cols]

        '''order columns by number of missing values in ascending order'''
        nr_missing = [dataframe[nan_col][dataframe[nan_col].isna()==True].shape[0] for nan_col in nan_cols]

        if order=='ascending':
            indice = np.argsort(nr_missing)
        elif order=='descending':
            indice = np.argsort(nr_missing)[::-1]
        elif order=='alphabetical':
            nan_cols_temp = self.nan_cols
            nan_cols_temp.sort()
            indice = [nan_cols_temp.index(col) for col in nan_cols_temp.columns.to_list()]
        elif order=='frame':
            indice = nr_missing
        elif type(order)==list:
            indice = [order.index(col) for col in self.nan_cols.columns.to_list()]
        else:
            print('Error (imputation order): give either imputation order')
        
        self.order = [self.nan_cols[j] for j in indice]
        dropped_cols = []

        if use_imputed == False:
            print('Warning: the imputed columns are not used for further imputations. Thus, any imputation order given is overruled. This setting is not recommended.')

        # begin loop through columns with missing data
        for nan_j in tqdm(self.order):

            # add column with missng data to dataframe according to specified order
            X[nan_j] = dataframe[nan_j]
            
            # define data without missing values in new column as train data
            X_train_data = X[X[nan_j].isna()==False].drop(nan_j, axis=1)
            y_train_data = X[X[nan_j].isna()==False][nan_j]

            # determine if column type is integer or float
            if y_train_data.dtype==float:
                # select regressor for flaot data
                xgb_j = xgb.XGBRegressor(tree_method='hist', enable_categorical=True)
                search_cv = cv
                scorer = 'neg_mean_absolute_error'
            else:
                # select classifier for integer data
                xgb_j = xgb.XGBClassifier(tree_method='hist', enable_categorical=True)
                search_cv = StratifiedKFold(n_splits=cv)
                scorer = accuracy_score #use no negative scoring because here the maximum is desired
            
            # determine which kind of search is specified and fit to train data
            if scheme == 'bayes':

                if params == None:
                    print('Error: for hyperparameter searches a parameter space must be chosen. Enter a dictionary for params.')
                    break

                search = BayesSearchCV(xgb_j, params, cv=search_cv, refit=True, scoring=scorer,
                                        n_jobs=-1, verbose=0, n_iter=nr_iterations)

                search.fit(X_train_data, y_train_data)
                self.models.append(search)
                
                self.best_scores.append(search.best_score_)
                self.best_parameters.append(search.best_params_)
                
                X_miss = X[X[nan_j].isna()==True].drop(nan_j, axis=1)
                y_miss = search.predict(X_miss)

            elif scheme == 'grid':

                if params == None:
                    print('Error: for hyperparameter searches a parameter space must be chosen. Enter a dictionary for params.')
                    break

                search = GridSearchCV(xgb_j, params, cv=cv, refit=True, scoring=scorer,
                                        n_jobs=-1, verbose=0, n_iter=nr_iterations)

                search.fit(X_train_data, y_train_data)
                self.models.append(search)
                
                self.best_scores.append(search.best_score_)
                self.best_parameters.append(search.best_params_)
                
                X_miss = X[X[nan_j].isna()==True].drop(nan_j, axis=1)
                y_miss = search.predict(X_miss)
            
            elif scheme == 'random':

                if params == None:
                    print('Error: for hyperparameter searches a parameter space must be chosen. Enter a dictionary for params.')
                    break

                search = RandomizedSearchCV(xgb_j, params, cv=cv, refit=True, scoring=scorer,
                                            n_jobs=-1, verbose=0, n_iter=nr_iterations)

                search.fit(X_train_data, y_train_data)
                self.models.append(search)
                
                self.best_scores.append(search.best_score_)
                self.best_parameters.append(search.best_params_)
                
                X_miss = X[X[nan_j].isna()==True].drop(nan_j, axis=1)
                y_miss = search.predict(X_miss)    

            elif scheme == 'default':

                if nr_iterations != 100:
                    print('Warning: number of iterations was entered but has no consequences in the default scheme.')
                xgb_j.fit(X_train_data, y_train_data)
                self.models.append(xgb_j)
                
                X_miss = X[X[nan_j].isna()==True].drop(nan_j, axis=1)
                y_miss = xgb_j.predict(X_miss)

            else:
                print('Error: no valid search scheme selected. Vaid schemes are bayes, grid, randomized and default')
                break
            

            '''glue predictions with non missing values together to construct column'''
            nan_count = 0
            for i in range(dataframe.shape[0]):
                if math.isnan(nan_col_data[nan_j].iloc[i]):
                    nan_col_data[nan_j].iloc[i] = y_miss[nan_count]
                    nan_count += 1
            
            # add new column with imputed values to dataframe
            X[nan_j] = nan_col_data[nan_j]
            
            # add column to use for further imputations if use_selected == True
            if use_imputed == False:
                dropped_cols.append(nan_j)

        self.df = X

    def fit(self, data):
        # copy fit_transform and do some adjustments
        pass
        
    def impute(self, data):

        for ind, col in enumerate(self.nan_cols):
            predictions = self.models[ind].predict(data[data[col].isna().any()])
            nan_count = 0
            for i in range(data.shape[0]):
                if math.isnan(data[col].iloc[i]):
                    data[col].iloc[i] = predictions[nan_count]
                    nan_count += 1
        return data