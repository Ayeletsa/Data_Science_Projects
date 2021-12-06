#!/usr/bin/env python
# coding: utf-8

# In[1]:


####################################################################################################
# wis_dnn_challenge.py
# Description: This is a template file for the WIS DNN challenge submission.
# Important: The only thing you should not change is the signature of the class (Predictor) and its predict function.
#            Anything else is for you to decide how to implement.
#            We provide you with a very basic working version of this class.
#
# Author: <first name1>_<last name1> [<first name1>_<last name2>]
#
# Python 3.7
####################################################################################################

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
from tensorflow.keras.callbacks import ModelCheckpoint


# The time series that you would get are such that the difference between two rows is 15 minutes.
# This is a global number that we used to prepare the data, so you would need it for different purposes.
DATA_RESOLUTION_MIN = 15


class Predictor(object):
    """
    This is where you should implement your predictor.
    The testing script calls the 'predict' function with the glucose and meals test data which you will need in order to
    build your features for prediction.
    You should implement this function as you wish, just do not change the function's signature (name, parameters).
    The other functions are here just as an example for you to have something to start with, you may implement whatever
    you wish however you see fit.
    """

    def __init__(self, path2data):
        """
        This constructor only gets the path to a folder where the training data frames are.
        :param path2data: a folder with your training data.
        """
        self.path2data = path2data
        self.train_glucose = None
        self.train_meals = None
        self.nn = None

    def predict(self, X_glucose, X_meals):
        """
        You must not change the signature of this function!!!
        You are given two data frames: glucose values and meals.
        For every timestamp (t) in X_glucose for which you have at least 12 hours (48 points) of past glucose and two
        hours (8 points) of future glucose, predict the difference in glucose values for the next 8 time stamps
        (t+15, t+30, ..., t+120).

        :param X_glucose: A pandas data frame holding the glucose values in the format you trained on.
        :param X_meals: A pandas data frame holding the meals data in the format you trained on.
        :return: A numpy ndarray, sized (M x 8) holding your predictions for every valid row in X_glucose.
                 M is the number of valid rows in X_glucose (number of time stamps for which you have at least 12 hours
                 of past glucose values and 2 hours of future glucose values.
                 Every row in your final ndarray should correspond to:
                 (glucose[t+15min]-glucose[t], glucose[t+30min]-glucose[t], ..., glucose[t+120min]-glucose[t])
        """

        # build features for set of (ID, timestamp)
        ids = self.cgm_and_meals_id_time.reset_index()['id'].unique()        
        x_all, y_all, id_all,x_GV= self.build_features(ids=ids)
        
        self.x_all = x_all
        self.y_all = y_all
        self.id_all = id_all
        self.x_GV = x_GV
        
        #load nn:
        self.load_nn_model()        
       
        # feed the network you trained
        y_predict = self.nn.predict([x_GV,x_all])
        
        self.y_predict = y_predict
        
        
        #creade data frame for true y with TA code        
        X_glucose2 = self.load_data_frame('GlucoseValues.csv')
        X, self.y_true  = self.build_features2(X_glucose2, None)

        # create df copy, sort and insert our prediction to create data frame with idx:
        y = self.y_true.copy()
        y = y.sort_values(['id','Date'])

        y['Glucose difference +15min'] = y_predict[:,0]
        y['Glucose difference +30min'] = y_predict[:,1]
        y['Glucose difference +45min'] = y_predict[:,2]
        y['Glucose difference +60min'] = y_predict[:,3]
        y['Glucose difference +75min'] = y_predict[:,4]
        y['Glucose difference +90min'] = y_predict[:,5]
        y['Glucose difference +105min'] = y_predict[:,6]
        y['Glucose difference +120min'] = y_predict[:,7]
        
        return y
    
    
    
    def load_nn_model(self):
        """
        Load your trained neural network.
        :return:
        """
        self.nn = tf.keras.models.load_model(self.path2data+'09-45-15.hdf5')
        pass

    @staticmethod
    def load_data_frame(path):
        """
        Load a pandas data frame in the relevant format.
        :param path: path to csv.
        :return: the loaded data frame.
        """
        return pd.read_csv(path, index_col=[0, 1], parse_dates=['Date'])

    def load_raw_data(self,train=False):
        """
        Loads raw data frames from csv files, and do some basic cleaning
        :return:
        """
        self.train_glucose = Predictor.load_data_frame(os.path.join(self.path2data, 'GlucoseValues.csv'))
        self.train_meals = Predictor.load_data_frame(os.path.join(self.path2data, 'Meals.csv'))

        self.X_glucose_df = self.train_glucose
        
        # remove food_id and unit_id columns from Meals df
        self.train_meals = self.train_meals.drop(['meal_type','unit_id','food_id'],axis=1)
       
       
        if train:
        # define  normaliation function, assumes an 'id' feature in dataframe!
            def normalize_df (df):
                mean_pd_by_id = df.groupby('id').mean()
                std_pd_by_id = df.groupby('id').std()

                df['Norm_GV'] = (df-mean_pd_by_id)/std_pd_by_id

                return df

            # normalize GV for every individual
            self.train_glucose = normalize_df (self.train_glucose)

            # remove outliers - thr std's above the mean
            thr_std_outlier = 4
            self.train_glucose = self.train_glucose[self.train_glucose['Norm_GV'].abs() < thr_std_outlier]

            # re-normalize after removing outliers
            self.train_glucose = normalize_df (self.train_glucose)
           
            #remove normalization:
            self.train_glucose= self.train_glucose.drop(['Norm_GV'],axis=1)
                 
            # remove outliers of meals - thr std's above the mean
            thr_std = 4
            thr_outlier = self.train_meals.copy().replace(0,np.nan)
            thr_outlier = thr_outlier.groupby('id').mean() + thr_std * thr_outlier.groupby('id').std()
            self.train_meals = self.train_meals[self.train_meals - thr_outlier < 0]
           
                  

        # resample meals using the timestamps from the glucose df
        timeStamp = str(DATA_RESOLUTION_MIN)+'T'
        self.train_meals = self.train_meals.groupby(pd.Grouper(level=0)).resample(timeStamp,level=-1).sum()

        # Normalize after removing outliers
        #self.train_meals = self.train_meals/self.train_meals.groupby('id').max()

        # resample glucose dataframe in timestamps identical to meals
        self.train_glucose = self.train_glucose.groupby(pd.Grouper(level=0)).resample(timeStamp,level=-1).first()        
       #find nan ind:
        
        self.train_glucose=self.train_glucose.dropna(how='any') 
        # replace NAN values in Glucose with linear interpolation
        #self.train_glucose = self.train_glucose.interpolate()
                
        # now merge the glucose and meals dataframes on the same time stamps
        self.cgm_and_meals_id_time = self.train_glucose.merge(self.train_meals,how='left',left_index=True,right_index=True)

        # replace NAN values in meals with 0's
        self.cgm_and_meals_id_time = self.cgm_and_meals_id_time.fillna(0)
        self.X_glucose=self.cgm_and_meals_id_time.iloc[:,0:1]
        self.X_meals=self.cgm_and_meals_id_time.iloc[:,1:]
     
        return self.X_glucose, self.X_meals
   
   
    def build_features(self,ids, build_y=False, n_previous_time_points=49):
        """
        Given glucose and meals data, build the features needed for prediction.
        :param X_glucose: A pandas data frame holding the glucose values.
        :param X_meals: A pandas data frame holding the meals data.
        :param build_y: Whether to also extract the values needed for prediction.
        :param n_previous_time_points:
        :return: The features needed for your prediction, and optionally also the relevant y arrays for training.
        """
        # function that create shifts:
        def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
            data = []
            labels = []
            start_index=0
            start_index = start_index + history_size-1
            if end_index is None:
                end_index = len(dataset) - target_size
     

            for i in range(start_index, end_index):
                indices = range(i-history_size+1, i+1, step)
                data.append(dataset[indices])

                if single_step:
                   labels.append(target[i+target_size])
                else:
                   labels.append(target[i+1:i+target_size+1]-target[i])

            return np.array(data), np.array(labels)

        # ,,,,,,,,,
        past_history = n_previous_time_points
        future_target = 8
        STEP = 1

   
        x_all = np.empty([1,past_history,self.cgm_and_meals_id_time.shape[1]])
        y_all = np.empty([1,future_target])
        id_all = np.empty([1,1])

        for id in ids:
            multi_data = self.cgm_and_meals_id_time.loc[id]
            end_index = len(multi_data)-future_target
            x_single, y_single = multivariate_data(multi_data.values, multi_data.GlucoseValue.values, 0,
                                                   len(multi_data.values)-future_target, past_history,
                                                   future_target, STEP)
                                                   
            x_all = np.append(x_all,x_single,axis=0)
            y_all = np.append(y_all,y_single,axis=0)
            id_all = np.append(id_all,id*np.ones([y_single.shape[0],1]),axis=0)
   
        # remove first empty example
        x_all = x_all[1:]
        y_all = y_all[1:]
        id_all = id_all[1:]
        x_GV=x_all[:,:,0:1]
       
        return x_all, y_all, id_all,x_GV

    def extract_y(df, n_future_time_points=8):
        """
        Extracting the m next time points (difference from time zero)
        :param n_future_time_points: number of future time points
        :return:
        """
        for g, i in zip(range(DATA_RESOLUTION_MIN, DATA_RESOLUTION_MIN*(n_future_time_points+1), DATA_RESOLUTION_MIN),
                        range(1, (n_future_time_points+1), 1)):
            df['Glucose difference +%0.1dmin' % g] = df.GlucoseValue.shift(-i) - df.GlucoseValue
        return df.dropna(how='any', axis=0).drop('GlucoseValue', axis=1)
    
    def create_shifts(df, n_previous_time_points=48):
        """
        Creating a data frame with columns corresponding to previous time points
        :param df: A pandas data frame
        :param n_previous_time_points: number of previous time points to shift
        :return:
        """
        for g, i in zip(range(DATA_RESOLUTION_MIN, DATA_RESOLUTION_MIN*(n_previous_time_points+1), DATA_RESOLUTION_MIN),
                        range(1, (n_previous_time_points+1), 1)):
            df['GlucoseValue -%0.1dmin' % g] = df.GlucoseValue.shift(i)
        return df.dropna(how='any', axis=0)

    def build_features2(self, X_glucose, X_meals, n_previous_time_points=48, n_future_time_points=8):
        """
        Given glucose and meals data, build the features needed for prediction.
        :param X_glucose: A pandas data frame holding the glucose values.
        :param X_meals: A pandas data frame holding the meals data.
        :param n_previous_time_points:
        :param n_future_time_points:
        :return: The features needed for your prediction, and optionally also the relevant y arrays for training.
        """
        # using X_glucose and X_meals to build the features
        # get the past 48 time points of the glucose
        X = X_glucose.reset_index().groupby('id').apply(Predictor.create_shifts, n_previous_time_points=n_previous_time_points).set_index(['id', 'Date'])
        # use the meals data...
        
        # this implementation of extracting y is a valid one.
        y = X_glucose.reset_index().groupby('id').apply(Predictor.extract_y, n_future_time_points=n_future_time_points).set_index(['id', 'Date'])
        index_intersection = X.index.intersection(y.index)
        X = X.loc[index_intersection]
        y = y.loc[index_intersection]
        return X, y


# In[2]:


if __name__ == "__main__":
    # example of predict() usage

    # create Predictor instance
    path2data = ''
    predictor = Predictor(path2data)

    # load the raw data
    X_glucose, X_meals=predictor.load_raw_data()
    y_predict=predictor.predict(X_glucose, X_meals)
 


# In[ ]:




