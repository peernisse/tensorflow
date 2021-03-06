#Installing
library(tensorflow)

#reticulate::conda_remove("r-reticulate")

#Install latest version
tensorflow::install_tensorflow()

#` to work with the tutorial need tf <v2
tensorflow::install_tensorflow(version = "1.13.1")

#Lesson 1 intro video

#Quiz 2
# Create your session
use_condaenv("r-reticulate")
sess <- tf$Session()

# Define a constant (you'll learn this next!)
HiThere <- tf$constant('Hi DataCamp Student!')

# Run your session with the HiThere constant
print(sess$run(HiThere))

# Close the session
sess$close()


#Syntax constants variables and placeholders
#Video
#https://campus.datacamp.com/courses/introduction-to-tensorflow-in-r/introducing-tensorflow-in-r?ex=4

#Constants
a<-tf$constant(2) #number data type dtype is from python is float32
class(a)

#Variables
#tf$Variable('initial value','optional name')
EmptyMatrix<- tf$Variable(tf$zeros(shape(4,3)))

#Placeholders
#'similar to variables but will assign
#'data at a later date
#'used when we know the shape of the tensor but will
#'use data from a previous pipeline execution
#'or external source
#'
#Example
#tf$placeholder(dtype,shape=None,name=None)

SinglePlaceholder<-tf$placeholder(tf$float32)

#Quiz
# Create two constant tensors
myfirstconstanttensor <- tf$constant(152)
mysecondconstanttensor <- tf$constant('I am a tensor master!')

print(sess$run(mysecondconstanttensor))

# Create a matrix of zeros
myfirstvariabletensor <- tf$Variable(tf$zeros(shape(5,1)))

sess$close()

#`Visualizing tensorflow models with tensorboard
#`tensorboard makes a visual map of data flow in the model
#`is browser based
#'Example with adults and children at picnic
library(tensorflow)
session<-tf$Session()

a<-tf$constant(5, name="NumAdults")
b<-tf$constant(6, name="NumChildren")
c<-tf$add(a,b)
print(session$run(c))

#'Open in tensorboard--this does not currently work
#Write to local machine
library(keras)
use_condaenv("r-reticulate")

writemygraph<-tf$summary$FileWriter('./graphs',session$graph)

writemygraph
tensorflow::tf_config()

callbacks = callback_tensorboard('./graphs')

tensorboard(log_dir='./graphs')

tensorflow::tf_config()

tensorboard()

#Practice





#STUFF-----------------------------------


old_path <- Sys.getenv("PATH")

Sys.setenv(PATH = paste(old_path, "C:\\Users\\peern\\AppData\\Local\\R-MINI~1\\Scripts", sep = ";"))



str(old_path)




