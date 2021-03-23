#Installing
install.packages("tensorflow")
library(tensorflow)
install_tensorflow()
Y

#` to work with the tutorial need tf <v2
tensorflow::install_tensorflow(version = "1.13.1")

#Lesson 1 intro video

#Quiz 2
# Create your session
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

# Create a matrix of zeros
myfirstvariabletensor <- tf$Variable(tf$zeros(shape(5,1)))

















