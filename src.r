# Required Library
#install.packages('reshape2')
#install.packages('lubridate')
library(reshape2)
library(lubridate)
library(foreach)
library(doSNOW)

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

# Function that implements the image algorithm
# @return: a prediction matrix
pred_algo_image <- function(train_set, test_set){
  
  # list the coordinates we have to predict
  coordinate.names <- gsub("_x", "", names(train_set)[grep("_x", names(train_set))])
  
  
  # for each one, compute the average patch
  mean.patches <- foreach(coord = coordinate.names) %dopar% {
    cat(sprintf("computing mean patch for %s\n", coord))
    coord_x <- paste(coord, "x", sep="_")
    coord_y <- paste(coord, "y", sep="_")
    
    # compute average patch
    patches <- foreach (i = 1:nrow(train_set), .combine=rbind) %do% {
      im  <- matrix(data = im.train[i,], nrow=96, ncol=96)
      x   <- train_set[i, coord_x]
      y   <- train_set[i, coord_y]
      x1  <- (x-patch_size)
      x2  <- (x+patch_size)
      y1  <- (y-patch_size)
      y2  <- (y+patch_size)
      if ( (!is.na(x)) && (!is.na(y)) && (x1>=1) && (x2<=96) && (y1>=1) && (y2<=96) )
      {
        as.vector(im[x1:x2, y1:y2])
      }
      else
      {
        NULL
      }
    }
    matrix(data = colMeans(patches), nrow=2*patch_size+1, ncol=2*patch_size+1)
  }
  
  # for each coordinate and for each test image, find the position that best correlates with the average patch
  p <- foreach(coord_i = 1:length(coordinate.names), .combine=cbind) %dopar% {
    # the coordinates we want to predict
    coord   <- coordinate.names[coord_i]
    coord_x <- paste(coord, "x", sep="_")
    coord_y <- paste(coord, "y", sep="_")
    
    # the average of them in the training set (our starting point)
    mean_x  <- mean(train_set[, coord_x], na.rm=T)
    mean_y  <- mean(train_set[, coord_y], na.rm=T)
    
    # search space: 'search_size' pixels centered on the average coordinates 
    x1 <- as.integer(mean_x)-search_size
    x2 <- as.integer(mean_x)+search_size
    y1 <- as.integer(mean_y)-search_size
    y2 <- as.integer(mean_y)+search_size
    
    # ensure we only consider patches completely inside the image
    x1 <- ifelse(x1-patch_size<1,  patch_size+1,  x1)
    y1 <- ifelse(y1-patch_size<1,  patch_size+1,  y1)
    x2 <- ifelse(x2+patch_size>96, 96-patch_size, x2)
    y2 <- ifelse(y2+patch_size>96, 96-patch_size, y2)
    
    # build a list of all positions to be tested
    params <- expand.grid(x = x1:x2, y = y1:y2)
    
    # for each image...
    r <- foreach(i = 1:nrow(test_set), .combine=rbind) %do% {
      if ((coord_i==1)&&((i %% 100)==0)) { cat(sprintf("%d/%d\n", i, nrow(test_set))) }
      im <- matrix(data = im.test[i,], nrow=96, ncol=96)
      
      # ... compute a score for each position ...
      r  <- foreach(j = 1:nrow(params), .combine=rbind) %do% {
        x     <- params$x[j]
        y     <- params$y[j]
        p     <- im[(x-patch_size):(x+patch_size), (y-patch_size):(y+patch_size)]
        score <- cor(as.vector(p), as.vector(mean.patches[[coord_i]]))
        score <- ifelse(is.na(score), 0, score)
        data.frame(x, y, score)
      }
      
      # ... and return the best
      best <- r[which.max(r$score), c("x", "y")]
    }
    names(r) <- c(coord_x, coord_y)
    r
  }
  
  # prepare file for submission
  predictions <- data.frame(ImageId = 1:nrow(test_set), p);
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
  return(sqrt(mean((test_set-pred_matrix)^2, na.rm=T)));
}


# Function that directly write train/validation/test dataset to the envoirnment
generate_dataset <- function(raw_data, train_ratio, validation_ratio, test_ration){
  rand_idx = sample(1:nrow(raw_data));
  
  idx.train <<- rand_idx[1:floor(train_ratio*nrow(raw_data))];
  idx.validation <<- rand_idx[ floor(validation_ratio*nrow(raw_data))+1 : floor(validation_ratio*nrow(raw_data))];
  idx.test <<- rand_idx[ floor(test_ration*nrow(raw_data))+1 : length(rand_idx)];
  
  
  d.train.train <<- raw_data[idx.train,];
  d.train.validation <<- raw_data[idx.validation,];
  d.train.test <<- raw_data[idx.test,];
}












###################################################################
######### EXECUTION SCRIPTS #######################################
###################################################################



# Initialization
set.seed(second(Sys.time()));
param.train.ratio = 0.6;
param.validation.ratio = 0.2;
param.test.ratio = 0.2;

# Naive Mean Method
generate_dataset(d.train, param.train.ratio, param.validation.ratio, param.test.ratio);
predictions.naive = pred_algo_naive(d.train.train, d.train.test);
head(predictions.naive)
# don't include the first image column
eval_prediction(predictions.naive[,2:ncol(predictions.naive)], d.train.test);
#create_submission(predictions.naive,"submission_apr_6.csv");

# Naive Image Algorithm
generate_dataset(d.train, param.train.ratio, param.validation.ratio, param.test.ratio);
predictions.image = pred_algo_image(d.train.train, d.train.test);
eval_prediction(predictions.image[,2:ncol(predictions.image)], d.train.test);


###################################################################
######### Visualization Code ######################################
###################################################################

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


