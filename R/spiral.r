# This script reads the train data and plots it

train <- read.csv("../extras/svm_train.csv");
test  <- read.csv("../extras/svm_test.csv");

train$color[train$Label==2] <- "red"
train$color[train$Label==1] <- "blue"
train$color[train$Label==0] <- "black"

plot(train$X, train$Y, col=train$color, main="Spiral Train", xlab="X", ylab="Y")



