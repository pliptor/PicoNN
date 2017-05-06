# Oscar Takeshita 2017
# This script reads the train data and plots it.
# Next, it makes SVM models with each of its kernels and plots the decision boundaries and areas.
#
# The train data is the spiral pattern used in the
# cn231 course by Andrej Karpathy

# The C++ code that generates the train and test patterns is located at 
# https://github.com/pliptor/PicoNN 

input_image  <- "input.png";
path <- "../extras/";

load_data <- function(file) {
	path <- "../extras/";
	if(!dir.exists(path)) {
		path <- "../input/"; # Changing to Kaggle environment
	}
	return (read.csv(paste0(path, file)));
};

train <- load_data("train.csv");
test  <- load_data("test.csv");
nclasses <- length(unique(train$Label));

# color list for each of the classes. The example has three classes but one extra is needed as required by filled.contour
clist      <- rainbow(nclasses + 1, s = 1,   v = 1, start = 0.2, 1, alpha = 1);
clistdesat <- rainbow(nclasses + 1, s = 0.5, v = 1, start = 0.2, 1, alpha = 1);

for (c in 1:nclasses) {
	train$color[train$Label==c-1] <- clist[c];
	train$colordesat[train$Label==c-1] <- clistdesat[c];
}

# plot train data
plot(train$X, train$Y, bg=train$color , pch=21,  main="Spiral Train", xlab="X", ylab="Y");

# write input image file 
if(dir.exists(path) && !file.exists(paste0(path, input_image))) {
	dev.copy(png, paste0(path, input_image));
	dev.off()
}

# transform Label column to factor so svm treats it as a categorical classification
train$Label <- factor(train$Label, levels=seq(0,nclasses,1), ordered=T);

# load svm library 
library(e1071);
svm_model <- function(title="SVM", kernel='radial') {
	model <- svm(Label ~ X + Y, data = train, kernel=kernel);
	print(model);
	summary(model);
	plot_title = paste(title,'kernel',kernel);

# predict and adjust offset so Labels return to the original set of {0, 1, 2}
	pred <- as.numeric(predict(model, test)) - 1;

# reshape prediction vector to a matrix to produce decision boundaries and areas
	z <- matrix(pred, nrow = sqrt(nrow(test)), byrow=F)
# Make a contour plot for the decision boundaries. Also called "level plot"
		filled.contour(x = seq(-1, 1, length.out = nrow(z)),
				y = seq(-1, 1, length.out = ncol(z)),
				z, levels = seq(-0.5, nclasses , 1), col = clistdesat,
				xlab="X", ylab="Y", main=plot_title, plot.axes = {points(train$X, train$Y, bg=train$color, pch=21); axis(1); axis(2) });
}

# default svm with radial basis kernel
svm_model(kernel='radial')
output_image <- "svmoutput.png";
# write output image file 
if(dir.exists(path) && !file.exists(paste0(path, output_image))) {
	dev.copy(png, paste0(path, output_image));
	dev.off()
}

# a linear kernel cannot handle the curved decision regions
svm_model(kernel='linear')
output_image <- "linearoutput.png";
# write output image file 
if(dir.exists(path) && !file.exists(paste0(path, output_image))) {
	dev.copy(png, paste0(path, output_image));
	dev.off()
}

# polynomial kernel has its own 'features' 
svm_model(kernel='polynomial')
output_image <- "polyoutput.png";
# write output image file 
if(dir.exists(path) && !file.exists(paste0(path, output_image))) {
	dev.copy(png, paste0(path, output_image));
	dev.off()
}

# sigmoid kernel results in an odd output. Where is the issue? 
svm_model(kernel='sigmoid')
output_image <- "sigmoidoutput.png";
# write output image file 
if(dir.exists(path) && !file.exists(paste0(path, output_image))) {
	dev.copy(png, paste0(path, output_image));
	dev.off()
}

