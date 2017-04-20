# This script reads the train data and plots it.
# Next, it makes an SVM model with radial basis kernel and plots the decision boundaries.
#
# The train data is the spiral pattern used in the
# cn231 course by Andrej Karpathy

# The C++ code that generates the train and test patterns is located at 
# https://github.com/pliptor/PicoNN 

input_image  <- "input.png";
output_image <- "svmoutput.png";
path <- "../extras/";

load_data <- function() {
	path <- "../extras/";
	if(!dir.exists(path)) {
		path <- "../input/"; #kaggle
	}
	assign("train", read.csv(paste0(path, "train.csv")), envir = .GlobalEnv);
	assign("test",  read.csv(paste0(path, "test.csv")),  envir = .GlobalEnv);
};

load_data();

train$color[train$Label==2] <- "darkred";
train$color[train$Label==1] <- "blue";
train$color[train$Label==0] <- "coral";

# plot train data
plot(train$X, train$Y, bg=train$color, pch=21,  main="Spiral Train", xlab="X", ylab="Y");

# write input image file 
if(dir.exists(path) && !file.exists(paste0(path, input_image))) {
	dev.copy(png, paste0(path, input_image));
	dev.off()
}

# transform Label column to factor so svm treats it as a categorical classification
train$Label <- factor(train$Label, levels=c("0","1","2"), ordered=T);

# load svm library 
library(e1071);
model <- svm(Label ~ ., data = train[,!colnames(train) %in% "color"]);
print(model);
summary(model);

# predict and adjust offset so Labels return to the original set of {0, 1, 2}
pred <- as.numeric(predict(model, test)) - 1;

z <- matrix(pred, nrow = 100, byrow=F)
# Make a contour plot for the decision boundaries. Also called "level plot"
filled.contour(x = seq(-1, 1, length.out = nrow(z)),
               y = seq(-1, 1, length.out = ncol(z)),
	       z, levels = seq(-0.5, 3 , 1), col = c("coral","blue","darkred","black"),
	       xlab="X", ylab="Y", main="SVM classification (kernel=radial basis)");

# write input image file 
if(dir.exists(path) && !file.exists(paste0(path, output_image))) {
	dev.copy(png, paste0(path, output_image));
	dev.off()
}
