#'Tutorial on tensorflow and keras
#'FOR TEXT CLASSIFICATION
#'https://tensorflow.rstudio.com/tutorials/beginners/basic-ml/tutorial_basic_text_classification/
#'
#'SEE ALSO: https://jjallaire.github.io/deep-learning-with-r-notebooks/notebooks/3.4-classifying-movie-reviews.nb.html

library(keras)
library(tidyverse)
library(pins)


#Download IMDB data--------------------
#Data source: https://www.kaggle.com/nltkdata/movie-review#movie_review.csv

#Use the pin package
#I downloaded a token file from my Kaggle profile saved in .json folder

#Register this token
board_register_kaggle(token = "C:\\Users\\peern\\OneDrive\\Documents\\.kaggle\\kaggle.json")


#Check what is on the data website API
paths <- pins::pin_get("nltkdata/movie-review", "kaggle")

# we only need the movie_review.csv file
path <- paths[1]

#Read in dataset
df<-readr::read_csv(path)

#Explore the data
head(df)
df %>% count(tag)
df$text[1]

#Split the data set into training and testing dfs, about 80/20 split

training_id<-sample.int(nrow(df), size = nrow(df)*0.8)#make vector of 1:nrow(df), then sample 80% out randomly

training<-df[training_id,]#Pull the rows identified in our random selection
testing<-df[-training_id,]#Pull rows not identified in our random selection

#Find distribution of number of words in each review
df$text %>% 
  strsplit(" ") %>% 
  sapply(length) %>% 
  summary() %>% 
  hist(breaks=20)

#Prepare the data-----------------------
#define our Text Vectorization layer, it will be 
#responsible to take the string input and convert it to a Tensor.

num_words <- 10000
max_length <- 50

text_vectorization <- layer_text_vectorization(
  max_tokens = num_words, 
  output_sequence_length = max_length, 
)

#Now, we need to adapt the Text Vectorization layer. It’s when we call adapt that the layer will 
#learn about unique words in our dataset and assign an integer value for each one.

text_vectorization %>% 
  adapt(df$text)

#We can now see the vocabulary is in our text vectorization layer.

# TODO see https://github.com/tensorflow/tensorflow/pull/34529

get_vocabulary(text_vectorization)

#ou can see how the text vectorization layer transforms it’s inputs:

text_vectorization(matrix(df$text[1], ncol = 1))

#Build the model----------------------------------------------

input <- layer_input(shape = c(1), dtype = "string")

output <- input %>% 
  text_vectorization() %>% 
  layer_embedding(input_dim = num_words + 1, output_dim = 16) %>%
  layer_global_average_pooling_1d() %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dropout(0.5) %>% 
  layer_dense(units = 1, activation = "sigmoid")

model <- keras_model(input, output)

#Compile model to set loss function and optimizer-----------------------------

model %>% compile(
  
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = list('accuracy')
  
)


#Train the model-------------------------

history <- model %>% fit(
  training$text,
  as.numeric(training$tag == "pos"),
  epochs = 5,
  batch_size = 512,
  validation_split = 0.2,
  verbose=2
)


#Evaluate the model---------------------------

results <- model %>% evaluate(testing$text, as.numeric(testing$tag == "pos"), verbose = 0)
results
#This model has about 67% accuracy and high loss at 0.642


#Create graph from the history object

plot(history)




#Predict some sentiment from the testing data set

dat<-text_vectorization(matrix(testing$text[1], ncol = 1))

testing$MODELED <- model %>% predict(testing$text) %>% round(.,0)

testing$BINARY<-as.numeric(testing$tag == "pos")

test_predictions[ , 1]


#Check how the predicted values align with the given train values
xxx<-test_df %>% mutate(PRED = test_predictions)

mdl<-lm(xxx$PRED~xxx$label)

plot(xxx$PRED~xxx$label)
abline(mdl)

summary(mdl)












