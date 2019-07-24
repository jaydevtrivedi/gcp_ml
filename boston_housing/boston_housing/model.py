#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers

# =============================================================================
# tf.logging.set_verbosity(v = tf.logging.INFO)
# =============================================================================
# =============================================================================
#     database_path="/Users/jaydevtrivedi/github/Datasets/boston_housing/housing.csv"
#     dependent_columns = ['RM', 'LSTAT', 'PTRATIO']
#     target_columns = ['MEDV']
#     train_size_value = 0.8
#     random_state_value = 0
#     input_dimensions = 3
#     batch_size = 100
#     steps_per_epoch = 1
#     validation_steps = 1
# =============================================================================

def get_dataset(database_path):
    dataset_csv = database_path
    dataset = pd.read_csv(dataset_csv)
    return dataset

def shuffle_split_data(dataset, dependent_columns, target_columns, train_size_value, random_state_value):
    dependent_variables = dataset[dependent_columns]
    target_variable = dataset[target_columns]
    
    # Shuffle and Split the data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(dependent_variables, target_variable, train_size=train_size_value, random_state=random_state_value)
    return X_train, X_test, y_train, y_test

def build_ann(input_dimensions):
    model = models.Sequential()
    model.add(layers.Dense(units=128, activation='relu', input_dim=input_dimensions))
    model.add(layers.Dense(units=64, activation='relu'))
    model.add(layers.Dense(units=1, activation='linear'))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model

def predict_some(regressor):
    predict_value = [6.575, 4.98, 15.3] # similar to the very first observation of the dataset
    predict_value = np.reshape(predict_value, (1,3))
    pd.DataFrame(predict_value).shape
    regressor.predict(predict_value)

def train_and_evaluate(hparams):
    
    database_path = hparams["database_path"]
    dependent_columns = ['RM', 'LSTAT', 'PTRATIO']
    target_columns = ['MEDV']
    train_size_value = hparams["train_size_value"]
    random_state_value = hparams["random_state_value"]
    input_dimensions = hparams["input_dimensions"]
    batch_size = hparams["batch_size"]
    epochs = hparams["epochs"]
    steps_per_epoch = hparams["steps_per_epoch"]
    validation_steps = hparams["validation_steps"]
    
    
    dataset = get_dataset(database_path)
    X_train, X_test, y_train, y_test = shuffle_split_data(dataset,
                                                          dependent_columns,
                                                          target_columns, 
                                                          train_size_value, 
                                                          random_state_value)
    
    regressor = build_ann(input_dimensions)
    history = regressor.fit(X_train, y_train, 
                            batch_size=batch_size, 
                            validation_data=(X_test, y_test),
                            epochs=epochs)

    predict_some(regressor)