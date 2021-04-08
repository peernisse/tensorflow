##'Tensorflow/Keras tutorials
#'SAVE AND RESTORE MODELS
#'https://tensorflow.rstudio.com/tutorials/beginners/basic-ml/tutorial_save_and_restore/
#

#Setup---------------------

#'We’ll use the MNIST dataset to train our model to demonstrate saving weights. 
#'To speed up these demonstration runs, only use the first 1000 examples:
#

library(keras)

mnist <- dataset_mnist()

c(train_images, train_labels) %<-% mnist$train
c(test_images, test_labels) %<-% mnist$test

#Just get first 1000
train_labels <- train_labels[1:1000]
test_labels <- test_labels[1:1000]

train_images <- train_images[1:1000, , ] %>%
  array_reshape(c(1000, 28 * 28))
train_images <- train_images / 255

test_images <- test_images[1:1000, , ] %>%
  array_reshape(c(1000, 28 * 28))
test_images <- test_images / 255

#Define a model as a function---------------------

# Returns a short sequential model
create_model <- function() {
  model <- keras_model_sequential() %>%
    layer_dense(units = 512, activation = "relu", input_shape = 784) %>%
    layer_dropout(0.2) %>%
    layer_dense(units = 10, activation = "softmax")
  model %>% compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics = list("accuracy")
  )
  model
}

model <- create_model()
summary(model)


#Save the entire model-----------------
#This saves so it can be used without access to the original code

#Saved model format

model <- create_model()

model %>% fit(train_images, train_labels, epochs = 5, verbose = 2)

#Save
model %>% save_model_tf("model")
list.files("model")

#Reload a saved model
new_model <- load_model_tf("model")
summary(new_model)



#HDF5 Format--------------------

model <- create_model()

model %>% fit(train_images, train_labels, epochs = 5, verbose = 2)

model %>% save_model_hdf5("my_model.h5")

new_model <- load_model_hdf5("my_model.h5")
summary(new_model)


#Saving custom objects (HDF5 format)-------------------

#see tutorial

#Save checkpoints during training---------------------

checkpoint_path <- "checkpoints/cp.ckpt"

# Create checkpoint callback
cp_callback <- callback_model_checkpoint(
  filepath = checkpoint_path,
  save_weights_only = TRUE,
  verbose = 0
)

model <- create_model()

model %>% fit(
  train_images,
  train_labels,
  epochs = 10, 
  validation_data = list(test_images, test_labels),
  callbacks = list(cp_callback),  # pass callback to training
  verbose = 2
)

list.files(dirname(checkpoint_path))


#'Now rebuild a fresh, untrained model,
#'and evaluate it on the test set. An untrained model will perform at chance levels (~10% accuracy):

fresh_model <- create_model()
fresh_model %>% evaluate(test_images, test_labels, verbose = 0)


#'Then load the weights from the latest checkpoint (epoch 10), and re-evaluate:

fresh_model %>% load_model_weights_tf(filepath = checkpoint_path)
fresh_model %>% evaluate(test_images, test_labels, verbose = 0)


#Checkpoint callback options------------------

#'Alternatively, you can decide to save only the best model, where best by default 
#'is defined as validation loss. 
#'See the documentation for callback_model_checkpoint for further information.

checkpoint_path <- "checkpoints/cp.ckpt"

# Create checkpoint callback
cp_callback <- callback_model_checkpoint(
  filepath = checkpoint_path,
  save_weights_only = TRUE,
  save_best_only = TRUE,
  verbose = 1
)

model <- create_model()

model %>% fit(
  train_images,
  train_labels,
  epochs = 10, 
  validation_data = list(test_images, test_labels),
  callbacks = list(cp_callback), # pass callback to training,
  verbose = 2
)

list.files(dirname(checkpoint_path))

#Manually save the weights--------------------

# Save the weights
model %>% save_model_weights_tf("checkpoints/cp.ckpt")

# Create a new model instance
new_model <- create_model()

# Restore the weights
new_model %>% load_model_weights_tf('checkpoints/cp.ckpt')

# Evaluate the model
new_model %>% evaluate(test_images, test_labels, verbose = 0)




































