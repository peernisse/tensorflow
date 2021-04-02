#'Tutorial on tensorflow and keras
#'FOR REGRESSION
#'https://tensorflow.rstudio.com/tutorials/beginners/basic-ml/tutorial_basic_regression/

library(keras)
library(tidyverse)
library(tfdatasets)

#Data Source---------------------------------------
#Boston housing prices
#https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html
#The boston housing data set, from 1978, is available from keras

#Get data set from web via keras command---------------
boston_housing<-dataset_boston_housing()

#Explore data set
str(boston_housing)

#Pull out the separate list
c(train_data,train_labels) %<-% boston_housing$train
c(test_data,test_labels) %<-% boston_housing$test

#Examples and features
paste0("Training entries: ", length(train_data), ", labels: ", length(train_labels))

#Each column is a variable described at the data source website

#Set column names for easier interp--------------------------
#From the website column names...
column_names <- c('CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
                  'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT')

#Make new tibbles with the new column names
train_df <- train_data %>% 
  as_tibble(.name_repair = "minimal") %>% 
  setNames(column_names) %>% 
  mutate(label = train_labels)

test_df <- test_data %>% 
  as_tibble(.name_repair = "minimal") %>% 
  setNames(column_names) %>% 
  mutate(label = test_labels)

#Examine a row of data
train_df[1,]

#Examine train labels--------------
train_labels[1:10] #Display first 10 entries

#Normalize features-------------------------
# It’s recommended to normalize features that use different scales and ranges. 
# Although the model might converge without feature normalization, it makes training more difficult, 
# and it makes the resulting model more dependent on the choice of units used in the input.
# 
# We are going to use the feature_spec interface implemented in the tfdatasets package for normalization. 
# The feature_columns interface allows for other common pre-processing operations on tabular data.

spec<-feature_spec(train_df, label ~ .) %>% 
  step_numeric_column(all_numeric(), normalizer_fn = scaler_standard()) %>% 
  fit()
  
spec























