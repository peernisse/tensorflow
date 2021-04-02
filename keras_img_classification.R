#This tutorial is the R Studio quickstart
#https://tensorflow.rstudio.com/tutorials/beginners/


#Load libraries
library(keras)


#Load and prepare the mnist dataset

mnist<-dataset_mnist() #http://yann.lecun.com/exdb/mnist/
class(mnist)
str(mnist)
str(mnist$train)

#The values of the pixels are between 1 and 250 need to normalize
mnist$train$x<-mnist$train$x/255
mnist$test$x<-mnist$test$x/255

#Define keras model
#Note that when using the Sequential API the first layer must 
#specify the input_shape argument which represents the dimensions of the input. In our case, images 28x28.

model<-keras_model_sequential() %>% 
  layer_flatten(input_shape = c(28,28)) %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(0.2) %>% 
  layer_dense(10, activation = 'softmax')

summary(model)


#Compile the model

model %>% 
  compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = "adam",
    metrics = "accuracy"
  )


#Fit the model with the fit function on the train dataset

model %>% 
  fit(
    x = mnist$train$x, y = mnist$train$y,
    epochs = 5,
    validation_split = 0.3,
    verbose = 2
  )

#Use model to make predictions from the test set

predictions <- predict(model, mnist$test$x)
head(predictions, 2)

#Access model performance

model %>% 
  evaluate(mnist$test$x, mnist$test$y, verbose = 0)

#Save keras model

save_model_tf(object = model, filepath = "model")

#Reload model and test it

reloaded_model <- load_model_tf("model")

all.equal(predict(model, mnist$test$x), predict(reloaded_model, mnist$test$x))

#Basic ML with keras------------------------------------


############################################################################
##################################################################
###########################################################


#Image classification example with fashion-mnist:https://github.com/zalandoresearch/fashion-mnist
#https://tensorflow.rstudio.com/tutorials/beginners/basic-ml/tutorial_basic_classification/

#Get data
library(keras)
library(tidyverse)
options(scipen=6)
fashion_mnist <- dataset_fashion_mnist()
class(fashion_mnist)
str(fashion_mnist)


#Pull data out into arrays
#The left pipe seems to separate the first order list items
c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test



#Store the class names for future use because in the dataset they are just numbers from 0-9
class_names = c('T-shirt/top',
                'Trouser',
                'Pullover',
                'Dress',
                'Coat', 
                'Sandal',
                'Shirt',
                'Sneaker',
                'Bag',
                'Ankle boot')

#Explore the dataset-----------------
dim(train_images)
dim(train_labels)
summary(train_labels)
train_labels[1:20]

dim(test_images)
dim(test_labels)

#preprocess the data-------------

image_1 <- as.data.frame(train_images[1, , ])
colnames(image_1) <- seq_len(ncol(image_1))
image_1$y <- seq_len(nrow(image_1))
image_1 <- gather(image_1, "x", "value", -y)
image_1$x <- as.integer(image_1$x)

ggplot(image_1, aes(x = x, y = y, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "black", na.value = NA) +
  scale_y_reverse() +
  theme_minimal() +
  theme(panel.grid = element_blank())   +
  theme(aspect.ratio = 1) +
  xlab("") +
  ylab("")


#Need to scale the pixel values to be between 0 and 1

train_images <- train_images / 255
test_images <- test_images / 255

#Display first 25 train images in a grid

par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
for (i in 1:25) { 
  
  img <- train_images[i, , ]
  img <- t(apply(img, 2, rev)) 
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste(class_names[train_labels[i] + 1]))
}


#Building the neural network requires configuring the layers of the model, then compiling the model.

#Set up the layers

model <- keras_model_sequential()
model %>%
  layer_flatten(input_shape = c(28, 28)) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')


#Compiling the model

#' Before the model is ready for training, it needs a few more settings. 
#' These are added during the model’s compile step:
#'   
#'   Loss function — This measures how accurate the model is during training. 
#'   We want to minimize this function to “steer” the model in the right direction.
#' 
#'   Optimizer — This is how the model is updated based on the data it sees and its loss function.
#'   
#'   Metrics —Used to monitor the training and testing steps. 
#'   The following example uses accuracy, the fraction of the images that are correctly classified.


model %>% compile(
  
  optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
  
)

#Train the model with the train dataset--------------

model %>% fit(train_images, train_labels, epochs = 5, verbose = 2)

#Evaluate accuracy----------------

score<-model %>% evaluate(test_images, test_labels, verbose = 2)

cat('Test loss:', score["loss"], "\n")

cat('Test accuracy:', score["accuracy"], "\n")


#Make predictions--------------------
#Now the model is trained (ie, fit and tested on the test data)

predictions<-model %>% predict(test_images)

#See what the first prediction thinks
predictions[1,]

#you can inspect the results to see highest accuracy or just ask
which.max(predictions[1,])
predictions[1,][10]
class_names[10]
#It says 10, but the numbers in the data set are zero based so the model thinks the image
#is object 9 -> Ankle Boot with around 

class_pred <- model %>% predict_classes(test_images)
class_pred[1:20]

test_labels[1]

#Let’s plot several images with their predictions. 
#Correct prediction labels are green and incorrect prediction labels are red.

par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
for (i in 1:25) { 
  img <- test_images[i, , ]
  img <- t(apply(img, 2, rev)) 
  # subtract 1 as labels go from 0 to 9
  predicted_label <- which.max(predictions[i, ]) - 1
  true_label <- test_labels[i]
  if (predicted_label == true_label) {
    color <- '#008800' 
  } else {
    color <- '#bb0000'
  }
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste0(class_names[predicted_label + 1], " (",
                      class_names[true_label + 1], ")"),
        col.main = color)
}


#Finally use the trained model to make a prediction about a single image

# Grab an image from the test dataset
# take care to keep the batch dimension, as this is expected by the model. Use drop=false to keep all dimensions?
img <- test_images[1, , , drop = FALSE]
dim(img)

#Predict the img
# subtract 1 as labels are 0-based
imgpredict<-model %>% predict(img)
imgpredict
which.max(imgpredict)-1




