# Oscar Takeshita 2017
# This script reads the predicted output and makes a plot with the decision boundaries.

title <- "PicoNN classification";
output_image <- "output.png";
path <- "../extras/";

load_data <- function(file) {
	path <- "../extras/";
	if(!dir.exists(path)) {
		path <- "../input/"; # chaning to kaggle environment
	}
	return (read.csv(paste0(path, file)));
};

train <- load_data("train.csv");
pred <- load_data("prediction.csv");

# color list for each of the classes. The example has three classes but one extra is needed as required by filled.contour
clist      <- rainbow(4, s = 1,   v = 1, start = 0.2, 1, alpha = 1);
clistdesat <- rainbow(4, s = 0.5, v = 1, start = 0.2, 1, alpha = 1);

for (c in 1:3) {
	train$color[train$Label==c-1] <- clist[c];
	train$colordesat[train$Label==c-1] <- clistdesat[c];
}


# reshape prediction vector to a matrix to produce decision boundaries and areas
z <- matrix(pred$Label, nrow = sqrt(nrow(pred)), byrow=F)
# Make a contour plot for the decision boundaries. Also called "level plot"
	filled.contour(x = seq(-1, 1, length.out = nrow(z)),
			y = seq(-1, 1, length.out = ncol(z)),
			z = z, levels = seq(-0.5, 3 , 1), col = clistdesat,
			xlab="X", ylab="Y", main=title, plot.axes = {points(train$X, train$Y, bg=train$color, pch=21); axis(1); axis(2) });

# write prediction image file 
if(dir.exists(path) && !file.exists(paste0(path, output_image))) {
	dev.copy(png, paste0(path, output_image));
	dev.off()
}
