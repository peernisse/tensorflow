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


# The spec created with tfdatasets can be used together with layer_dense_features to 
# perform pre-processing directly in the TensorFlow graph.
# 
# We can take a look at the output of a dense-features layer created by this spec:

layer<-layer_dense_features(
  feature_columns = dense_features(spec),
  dtype = tf$float32
)

layer(train_df)


#Create the Model-----------------------

input<-layer_input_from_dataset(train_df %>% select(-label))

output<-input %>% 
  layer_dense_features(dense_features(spec)) %>% 
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1) 

model <- keras_model(input, output)

summary(model)

#Compile the model-------------------------------------

model %>% 
  compile(
    loss = 'mse',
    optimizer = optimizer_rmsprop(),
    metrics = list('mean_absolute_error')
  )


#Wrap the model build code as a function for future use on multiple experiments

build_model <- function() {
  input <- layer_input_from_dataset(train_df %>% select(-label))
  
  output <- input %>% 
    layer_dense_features(dense_features(spec)) %>% 
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 1) 
  
  model <- keras_model(input, output)
  
  model %>% 
    compile(
      loss = "mse",
      optimizer = optimizer_rmsprop(),
      metrics = list("mean_absolute_error")
    )
  
  model
}

#Train the model-------------------------------
#Create a dot progress meter

# Display training progress by printing a single dot for each completed epoch.
print_dot_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs) {
    if (epoch %% 80 == 0) cat("\n")
    cat(".")
  }
)    

model <- build_model()

#Remember that Keras fit modifies the model in-place.

history <- model %>% fit(
  x = train_df %>% select(-label),
  y = train_df$label,
  epochs = 500,
  validation_split = 0.2,
  verbose = 0,
  callbacks = list(print_dot_callback)
)


#Visualize the model training progress in ggplot2 from the history file we just made

plot(history)


# This graph shows little improvement in the model after about 200 epochs. 
# Let’s update the fit method to automatically stop training when the validation score doesn’t improve. 
# We’ll use a callback that tests a training condition for every epoch. 
# If a set amount of epochs elapses without showing improvement, it automatically stops the training.

# The patience parameter is the amount of epochs to check for improvement.
#Early stopping is a tecnique to prevent over fitting

early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)

model <- build_model()

history <- model %>% fit(
  x = train_df %>% select(-label),
  y = train_df$label,
  epochs = 500,
  validation_split = 0.2,
  verbose = 0,
  callbacks = list(early_stop)
)

plot(history)

#The MSE graph shows the average error is about $2,500. Is this good?
#Well,$2,500 is not an insignificant amount when some of the labels are only $15,000.

#See how the model performs on the test data set---------------------

c(loss, mae) %<-% (model %>% evaluate(test_df %>% select(-label), test_df$label, verbose = 0))

paste0("Mean absolute error on test set: $", sprintf("%.2f", mae * 1000))


# Use model to make a prediction---------------------------

#Predict some housing prices from the testing data set

test_predictions <- model %>% predict(test_df %>% select(-label))

test_predictions[ , 1]


#Check how the predicted values align with the given train values
xxx<-test_df %>% mutate(PRED = test_predictions)

mdl<-lm(xxx$PRED~xxx$label)

plot(xxx$PRED~xxx$label)
abline(mdl)

summary(mdl)












