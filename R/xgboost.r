# Oscar Takeshita 2017
# This script reads the train data and plots it.
# Next, it makes an xgboost model and plots the decision boundaries and areas.
#
# The train data is the spiral pattern used in the
# cn231 course by Andrej Karpathy

# The C++ code that generates the train and test patterns is located at 
# https://github.com/pliptor/PicoNN 

input_image  <- "input.png";
path <- "../extras/";

load_data <- function(file) {
	if(!dir.exists(path)) {
		path <- "../input/"; # chaning to kaggle environment
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
	dev.off();
}

# load xgboost library 
library("xgboost");
library("caret")    # for the confusionmatrix() function (also needs e1071 package)
library("dplyr")    # for some data preperation

# xgboost needs data to be in its xgb.DMatrix format
# we prepare a split of the data for confusion matrix computation

# Make split index
dat <- train[,1:3];
train_index <- sample(1:nrow(dat), nrow(dat)*0.80);
# Full data set
data_variables <- as.matrix(dat[,2:3]);
data_label <- dat[,"Label"];
data_matrix <- xgb.DMatrix(data = as.matrix(dat), label = data_label);
# split train data and make xgb.DMatrix
train_data   <- data_variables[train_index,];
train_label  <- data_label[train_index];
train_matrix <- xgb.DMatrix(data = train_data, label = train_label);
# split test data and make xgb.DMatrix
test_data  <- data_variables[-train_index,];
test_label <- data_label[-train_index];
test_matrix <- xgb.DMatrix(data = test_data, label = test_label);

xgb_params <- list("objective" = "multi:softprob",
		"eval_metric" = "mlogloss",
		"num_class" = nclasses);
nround    <- 100; # number of XGBoost rounds
cv.nfold  <- 10; 
# Fit cv.nfold * cv.nround XGB models and save OOF predictions

cv_model <- xgb.cv(params = xgb_params,
		data = train_matrix, 
		nrounds = nround,
		nfold = cv.nfold,
		verbose = F,
		prediction = TRUE);

OOF_prediction <- data.frame(cv_model$pred) %>%
mutate(max_prob = max.col(., ties.method = "last"),
		label = train_label + 1);
head(OOF_prediction);

# confusion matrix
cm <- confusionMatrix(factor(OOF_prediction$label), 
		factor(OOF_prediction$max_prob),
		mode = "everything");

bst_model <- xgb.train(params = xgb_params,
		data = train_matrix,
		nrounds = nround);

# Predict hold-out test set
test_pred <- predict(bst_model, newdata = test_matrix);
test_prediction <- matrix(test_pred, nrow = nclasses,
		ncol=length(test_pred)/nclasses) %>%
t() %>%
data.frame() %>%
mutate(label = test_label + 1,
		max_prob = max.col(., "last"));
# confusion matrix of test set
ans <- confusionMatrix(factor(test_prediction$label),
		factor(test_prediction$max_prob),
		mode = "everything");

# predict on the test grid
test_label <- rep(0, nrow(test)); 
test_pred <- predict(bst_model, newdata = xgb.DMatrix(as.matrix(test)));
Prediction <- matrix(test_pred, nrow = nclasses,
		ncol=length(test_pred)/nclasses) %>%
t() %>%
data.frame() %>%
mutate(label = test_label + 1,
		max_prob = max.col(., "last"));

# reshape prediction vector to a matrix to produce decision boundaries and areas
z <- matrix(Prediction$max_prob-1, nrow = sqrt(nrow(test)), byrow=F);
# Make a contour plot for the decision boundaries. Also called "level plot"
filled.contour(x = seq(-1, 1, length.out = nrow(z)),
		y = seq(-1, 1, length.out = ncol(z)),
		z, levels = seq(-0.5, 3 , 1), col = clistdesat,
		xlab="X", ylab="Y", main="xgboost", plot.axes = {points(train$X, train$Y, bg=train$color, pch=21); axis(1); axis(2) });

output_image <- paste0("xgboost",".png");
if(!dir.exists(path)) {
	path <- "../input/"; # chaning to kaggle environment
}
# write output image file 
if(dir.exists(path) && !file.exists(paste0(path, output_image))) {
	dev.copy(png, paste0(path, output_image));
	dev.off()
} else {
	dev.copy(png, output_image);
	dev.off()
}
