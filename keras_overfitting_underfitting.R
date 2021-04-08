#'Tensorflow/Keras tutorials
#'UNDERFITTING AND OVERFITTING
#'https://tensorflow.rstudio.com/tutorials/beginners/basic-ml/tutorial_overfit_underfit/
#'
#In this tutorial, we’ll explore two common regularization techniques — 
#weight regularization and dropout — and use them to improve our IMDB movie review classification results.

library(keras)
library(tidyverse)

#Download the IMDB data set

num_words<-1000
imdb<-dataset_imdb(num_words = num_words)

c(train_data, train_labels) %<-% imdb$train
c(test_data, test_labels) %<-% imdb$test

#'Rather than using an embedding as in the previous notebook, here we will multi-hot encode the sentences. 
#'This model will quickly overfit to the training set. It will be used to demonstrate 
#'when overfitting occurs, and how to fight it.
#'Multi-hot-encoding our lists means turning them into vectors of 0s and 1s.
#'Concretely, this would mean for instance turning the sequence [3, 5] into a 10,000-dimensional 
#'vector that would be all-zeros except for indices 3 and 5, which would be ones.

#Make multi-hot function
multi_hot_sequences<-function(sequences, dimension){
  
  multi_hot<-matrix(0,nrow = length(sequences), ncol = dimension)
  
  for(i in 1:length(sequences)){
    multi_hot[i, sequences[[i]]] <- 1
  }#end for
  
  multi_hot
  
}#end multi_hot_sequences


#Apply the encoding function to the data sets
train_data<-multi_hot_sequences(train_data, num_words)
test_data<-multi_hot_sequences(test_data, num_words)

#'Let’s look at one of the resulting multi-hot vectors. The word indices are sorted by frequency, 
#'so it is expected that there are more 1-values near index zero, as we can see in this plot:

first_text <- data.frame(word = 1:num_words, value = train_data[1, ])

ggplot(first_text, aes(x = word, y = value)) +
  geom_line() +
  theme(axis.title.y = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank())


#Create a baseline model

baseline_model <- 
  keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = num_words) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

#Compile model to optimize
baseline_model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = list("accuracy")
)

summary(baseline_model)


#Fit model to train data and store output in object history

baseline_history <- baseline_model %>% fit(
  train_data,
  train_labels,
  epochs = 20,
  batch_size = 512,
  validation_data = list(test_data, test_labels),
  verbose = 2
)

#Gets overfit pretty quick


#Create a smaller model with fewer hidden units to compare to the baseline model

smaller_model <- 
  keras_model_sequential() %>%
  layer_dense(units = 4, activation = "relu", input_shape = num_words) %>%
  layer_dense(units = 4, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

#Compile to optimize
smaller_model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = list("accuracy")
)

summary(smaller_model)

#Train the model using the same data

smaller_history <- smaller_model %>% fit(
  train_data,
  train_labels,
  epochs = 20,
  batch_size = 512,
  validation_data = list(test_data, test_labels),
  verbose = 2
)


#Create a bigger model to compare. This model has much bigger capcity than needed

bigger_model <- 
  keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu", input_shape = num_words) %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

#Compile
bigger_model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = list("accuracy")
)

summary(bigger_model)

#Train the model using the same data

bigger_history <- bigger_model %>% fit(
  train_data,
  train_labels,
  epochs = 20,
  batch_size = 512,
  validation_data = list(test_data, test_labels),
  verbose = 2
)


#Plot the training and validation loss---------------
compare_cx <- data.frame(
  baseline_train = baseline_history$metrics$loss,
  baseline_val = baseline_history$metrics$val_loss,
  smaller_train = smaller_history$metrics$loss,
  smaller_val = smaller_history$metrics$val_loss,
  bigger_train = bigger_history$metrics$loss,
  bigger_val = bigger_history$metrics$val_loss
) %>%
  rownames_to_column() %>%
  mutate(rowname = as.integer(rowname)) %>%
  gather(key = "type", value = "value", -rowname)

ggplot(compare_cx, aes(x = rowname, y = value, color = type)) +
  geom_line() +
  xlab("epoch") +
  ylab("loss")


#Evaluate models
results_baseline<-baseline_model %>% evaluate(test_data,test_labels)
results_baseline

results_smaller<-smaller_model %>% evaluate(test_data,test_labels)
results_smaller

results_bigger<-bigger_model %>% evaluate(test_data,test_labels)
results_bigger

#Strategies--------------------

#Add weight regularization

l2_model <- 
  keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = num_words,
              kernel_regularizer = regularizer_l2(l = 0.001)) %>%
  layer_dense(units = 16, activation = "relu",
              kernel_regularizer = regularizer_l2(l = 0.001)) %>%
  layer_dense(units = 1, activation = "sigmoid")

l2_model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = list("accuracy")
)

l2_history <- l2_model %>% fit(
  train_data,
  train_labels,
  epochs = 20,
  batch_size = 512,
  validation_data = list(test_data, test_labels),
  verbose = 2
)

compare_cx <- data.frame(
  baseline_train = baseline_history$metrics$loss,
  baseline_val = baseline_history$metrics$val_loss,
  l2_train = l2_history$metrics$loss,
  l2_val = l2_history$metrics$val_loss
) %>%
  rownames_to_column() %>%
  mutate(rowname = as.integer(rowname)) %>%
  gather(key = "type", value = "value", -rowname)

ggplot(compare_cx, aes(x = rowname, y = value, color = type)) +
  geom_line() +
  xlab("epoch") +
  ylab("loss")


#Add Dropout

dropout_model <- 
  keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = num_words) %>%
  layer_dropout(0.6) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dropout(0.6) %>%
  layer_dense(units = 1, activation = "sigmoid")

dropout_model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = list("accuracy")
)

dropout_history <- dropout_model %>% fit(
  train_data,
  train_labels,
  epochs = 20,
  batch_size = 512,
  validation_data = list(test_data, test_labels),
  verbose = 2
)

compare_cx <- data.frame(
  baseline_train = baseline_history$metrics$loss,
  baseline_val = baseline_history$metrics$val_loss,
  dropout_train = dropout_history$metrics$loss,
  dropout_val = dropout_history$metrics$val_loss
) %>%
  rownames_to_column() %>%
  mutate(rowname = as.integer(rowname)) %>%
  gather(key = "type", value = "value", -rowname)

ggplot(compare_cx, aes(x = rowname, y = value, color = type)) +
  geom_line() +
  xlab("epoch") +
  ylab("loss")



#Evaluate models
results_l2<-l2_model %>% evaluate(test_data,test_labels)
results_l2

results_dropout <- dropout_model %>% evaluate(test_data,test_labels)
results_dropout












