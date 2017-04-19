# This script reads the train data and plots it.
# The train data is the spiral pattern used in the
# cn231 course by Andrej Karpathy

# The C++ code that generates it is located in
# src/piconn.cpp 

train <- read.csv("../extras/train.csv");
test  <- read.csv("../extras/test.csv");

train$color[train$Label==2] <- "red"
train$color[train$Label==1] <- "blue"
train$color[train$Label==0] <- "black"

plot(train$X, train$Y, col=train$color, main="Spiral Train", xlab="X", ylab="Y")



