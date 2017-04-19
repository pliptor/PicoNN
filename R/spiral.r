# This script reads the train data and plots it.
# The train data is the spiral pattern used in the
# cn231 course by Andrej Karpathy

# The C++ code that generates it is located at 
# https://github.com/pliptor/PicoNN 


load_data <- function() {
	path <- "../extras/";        #github repo
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

plot(train$X, train$Y, bg=train$color, pch=21,  main="Spiral Train", xlab="X", ylab="Y");

# write file 
if(dir.exists("../extras/")) {
	dev.copy(png,"../extras/input.png");
	dev.off()
}



