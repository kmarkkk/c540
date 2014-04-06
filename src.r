# Required Library
#install.packages('reshape2')
#install.packages('lubridate')
library(reshape2)
library(lubridate)

###################################################################
######### SET UP ENVIRONMENT ######################################
###################################################################

# Load enviornment
data.dir="~/Academy/2014 SPRING/COMP 540/Term Project/Facial Expression/data/";
setwd(data.dir);
load('data.Rd');


###################################################################
######### HELEPER FUNCTIONS #######################################
###################################################################


# Function that implements the naive algorithm that returns the average of the keypoints
# @return: a prediction matrix
pred_algo_naive <- function(train_set, test_set){
  p <- matrix(data=colMeans(train_set, na.rm=T), nrow=nrow(test_set), ncol=ncol(train_set), byrow=T)
  colnames(p) <- names(train_set)
  predictions <- data.frame(ImageId = 1:nrow(test_set), p)
  return(predictions);
}


# Function that creates a csv file that is submittable
create_submission <- function(pred_matrix, file_name){
  submission <- melt(pred_matrix, id.vars="ImageId", variable.name="FeatureName", value.name="Location")
  example.submission <- read.csv(paste0(data.dir, 'IdLookupTable.csv'))
  sub.col.names = c("RowId","Location")
  example.submission$Location <- NULL
  submission <- merge(example.submission, submission, all.x=T, sort=F)
  submission <- submission[, sub.col.names]
  write.csv(submission, file=file_name, quote=F, row.names=F)  
}

# Function that scores a prediction
# @return: a double that denotes the score
eval_prediction <- function(pred_matrix, test_set){
  cat(nrow(pred_matrix),",",ncol(pred_matrix),"\n");
  cat(nrow(test_set),",",ncol(test_set),"\n");
  return(sqrt(mean((test_set-pred_matrix)^2, na.rm=T)));
}




###################################################################
######### EXECUTION SCRIPTS #######################################
###################################################################




# Divide Data: trainning, validation, test
param.train.ratio = 0.6;
param.validation.ratio = 0.2;
param.test.ratio = 0.2;
set.seed(second(Sys.time()));
rand_idx = sample(1:nrow(d.test));
idx.train = rand_idx[1:floor(param.train.ratio*nrow(d.test))];
idx.validation = rand_idx[ floor(param.train.ratio*nrow(d.test))+1 : floor(param.validation.ratio*nrow(d.test))];
idx.test = rand_idx[ floor(param.validation.ratio*nrow(d.test))+1 : length(rand_idx)];
d.train.train = d.train[idx.train,];
d.train.validation = d.train[idx.validation,];
d.train.test = d.train[idx.test,];



# Visualize the image
im <- matrix(data=rev(im.train[1,]), nrow=96, ncol=96);
image(1:96, 1:96, im, col=gray((0:255)/255));
# vis key points
points(96-d.train$nose_tip_x[1],         96-d.train$nose_tip_y[1],         col="red");
points(96-d.train$left_eye_center_x[1],  96-d.train$left_eye_center_y[1],  col="blue");
points(96-d.train$right_eye_center_x[1], 96-d.train$right_eye_center_y[1], col="green");

for(i in 1:nrow(d.train)) {
  points(96-d.train$nose_tip_x[i], 96-d.train$nose_tip_y[i], col="red")
}

idx <- which.max(d.train$nose_tip_x)
im  <- matrix(data=rev(im.train[idx,]), nrow=96, ncol=96)
image(1:96, 1:96, im, col=gray((0:255)/255))
points(96-d.train$nose_tip_x[idx], 96-d.train$nose_tip_y[idx], col="red")


# Naive Mean Method
predictions = pred_algo_naive(d.train.train, d.train.test);
head(predictions)
# don't include the first image column
eval_prediction(predictions[,2:ncol(predictions)], d.train.test);




