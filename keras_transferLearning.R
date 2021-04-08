##'Tutorial on tensorflow and keras
#'FOR TRANSFER LEARNING
#'https://tensorflow.rstudio.com/tutorials/beginners/basic-ml/tutorial_basic_text_classification_with_tfhub/
#'
#'

#Install tensorflow datasets and python module
#install.packages("remotes")
#remotes::install_github("rstudio/tfds", force=T)

#Had to install tf_datasets via conda install. There was a circular error
#where tfds could not load until the datasets were loaded
reticulate::conda_install('r-reticulate',"tensorflow_datasets")

#Then this worked OK
tfds::install_tfds()

install_tensorflow(envname = 'r-reticulate')

library(tensorflow)
library(reticulate)
library(keras)
library(tfhub)
library(tfds)
library(tfdatasets)


#Download the IMDB dataset
imdb<-tfds_load(
  "imdb_reviews:1.0.0",
  split = list("train[:60%]","train[-40%:]", "test"),
  as_supervised = TRUE
)

summary(imdb)








