# Required Library
#install.packages('reshape2')
#install.packages('lubridate')
#source("http://bioconductor.org/biocLite.R")
#biocLite("EBImage")
#install.packages('biOps');

library(reshape2)
library(lubridate)
library(foreach)
library(ggplot2)
#library(doSNOW)
#library(EBImage)
#library(biOps)

###################################################################
######### SET UP ENVIRONMENT ######################################
###################################################################

# Load enviornment
data.dir="/Users/wenxuancai/CS/540/c540";
setwd(data.dir);
load('data.Rd');
feature.train <-  read.csv('train_feature.csv');
d.train$id = 1:nrow(d.train);
d.train = cbind(d.train, feature.train);
rm(feature.train);
rm(d.test)
#feature.train.sub <- read.csv('train_feature_subset.csv');
feature.test <- read.csv('test_feature.csv');

patch_size  <- 10
search_size <- 2

# Data Clean-up
# Method 1: substitute na with column mean
#remove_na <- function(x){
#  x = as.numeric(x);
#  x[is.na(x)]=mean(x, na.rm=T);
#  return(x);
#}
#d.train = data.frame(apply(d.train, 2, remove_na))

# Method 2: filter out row with na
#d.train = na.omit(d.train)

#write.csv(im.test, file = "test_img.csv", row.names = FALSE);
#write.csv(im.train[1:100,], file = "train_img_subset.csv", row.names = FALSE);

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
pred_algo_image <- function(train_set, train_img, test_set, test_img) {
  
  # list the coordinates we have to predict
  coordinate.names <- gsub("_x", "", names(train_set)[grep("_x", names(train_set))])
  
  
  # for each one, compute the average patch
  mean.patches <- foreach(coord = coordinate.names) %do% {
    cat(sprintf("computing mean patch for %s\n", coord))
    coord_x <- paste(coord, "x", sep="_")
    coord_y <- paste(coord, "y", sep="_")
    
    # compute average patch
    patches <- foreach (i = 1:nrow(train_set), .combine=rbind) %do% {
      im  <- matrix(data = train_img[i,], nrow=96, ncol=96)
      x   <- train_set[i, coord_x]
      y   <- train_set[i, coord_y]
      x1  <- (x - patch_size)
      x2  <- (x + patch_size)
      y1  <- (y - patch_size)
      y2  <- (y + patch_size)
      if ( (!is.na(x)) && (!is.na(y)) && (x1>=1) && (x2<=96) && (y1>=1) && (y2<=96) )
      {
        as.vector(im[x1:x2, y1:y2])
      }
      else
      {
        NULL
      }
    }
    matrix(data = colMeans(patches), nrow=2 * patch_size + 1, ncol=2 * patch_size + 1)
  }
  
  # for each coordinate and for each test image, find the position that best correlates with the average patch
  p <- foreach(coord_i = 1:length(coordinate.names), .combine=cbind) %dopar% {
    # the coordinates we want to predict
    coord   <- coordinate.names[coord_i]
    coord_x <- paste(coord, "x", sep="_")
    coord_y <- paste(coord, "y", sep="_")
    
  
    # for each image...
    r <- foreach(i = 1:nrow(test_set), .combine=rbind) %do% {
      if ((coord_i==1)&&((i %% 100)==0)) { cat(sprintf("%d/%d\n", i, nrow(test_set))) }
      
      # the average of them in the training set (our starting point)
      start_x  <- test_set[i,][coord_x];
      start_y  <- test_set[i,][coord_y];
      
      # search space: 'search_size' pixels centered on the average coordinates 
      x1 <- as.integer(start_x) - search_size
      x2 <- as.integer(start_x) + search_size
      y1 <- as.integer(start_y) - search_size
      y2 <- as.integer(start_y) + search_size
      
      # ensure we only consider patches completely inside the image
      x1 <- ifelse(x1-patch_size<1,  patch_size+1,  x1)
      y1 <- ifelse(y1-patch_size<1,  patch_size+1,  y1)
      x2 <- ifelse(x2+patch_size>96, 96-patch_size, x2)
      y2 <- ifelse(y2+patch_size>96, 96-patch_size, y2)
      
      # build a list of all positions to be tested
      params <- expand.grid(x = x1:x2, y = y1:y2)
      
      im <- matrix(data = test_img[i,], nrow=96, ncol=96)
      
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



# Function that uses linear regression to model the problem
pred_algo_lm <- function(train_set, test_feature){
  
  # Train a lm model for each column
  coordinate.names <- names(train_set);
  p <- matrix(0, nrow(test_feature), 30)
  for (i in 1:30){
    # construct formula
    #f = as.formula(paste( coordinate.names[i], '~', paste(colnames(train_set)[31:37] , collapse='+')))
    
    f = as.formula(paste( coordinate.names[i], '~', 'centroid_x+centroid_y+ellipse_center_x+ellipse_center_y'))
    # train model
    lm.models = lm(f, data=train_set);
    
    # predict
    p[,i] = predict(lm.models, test_feature);

  }
  
  
  # Construct the output
  colnames(p) <- names(train_set)[1:30]
  predictions <- data.frame(ImageId = 1:nrow(test_feature), p)
  return(predictions)
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
generate_dataset <- function(raw_data, image_data, train_ratio, validation_ratio, test_ration){
  rand_idx = sample(1:nrow(raw_data));
  test.idx.end = floor(train_ratio*nrow(raw_data));
  validation.idx.end =  test.idx.end + 1 + floor(validation_ratio*nrow(raw_data));
  
  idx.train <<- rand_idx[1:test.idx.end];
  idx.validation <<- rand_idx[ (test.idx.end+1) :validation.idx.end];
  idx.test <<- rand_idx[ (validation.idx.end+1) :length(rand_idx)];

  
  d.train.train <<- raw_data[idx.train,];
  im.train.train <<- image_data[idx.train,];
  d.train.validation <<- raw_data[idx.validation,];
  im.train.validation <<- image_data[idx.validation,];
  d.train.test <<- raw_data[idx.test,];
  im.train.test <<- image_data[idx.test,]
}


# Function that visualize the results
vis_img <- function(image_set, train_set,prediction){
  
  coordinate.names <- gsub("_x", "", names(train_set)[grep("_x", names(d.train))]);
  
  for (i in 1:nrow(image_set)){
    im <- matrix(data=rev(image_set[i,]), nrow=96, ncol=96);
    image(1:96, 1:96, im, col=gray((0:255)/255));
    # add points
    for (j in 1:length(coordinate.names)) {
      coord_x <- paste(coordinate.names[j], "x", sep="_");
      coord_y <- paste(coordinate.names[j], "y", sep="_");
      
      points(96-prediction[i,coord_x], 96-prediction[i,coord_y], col="red")
      points(96-train_set[i,coord_x], 96-train_set[i,coord_y], col="blue")
      
    }
    title(paste(train_set[i,]$id));
    line = readline();
    cat(train_set[i,]$id,"\n");
    if (line == 'break') {
      break;
    }
    
  } 
}


# Draw the learning curve
leanring_curve <- function(interval, train_set, test_set, FUN = algo){
  
  sq_error = 1:length(interval);
  for (i in 1:length(interval)) {
    pred = FUN(train_set[1:interval[i],], test_set[,32:38]);
    
    sq_error[i] = eval_prediction(pred[,2:31], test_set[,1:30]);
  }
  
  df = data.frame(train_size = interval, error = sq_error )
  ggplot(df, aes(x = train_size, y = error)) + geom_line()+ggtitle("Learning Curve");
}





###################################################################
######### EXECUTION SCRIPTS #######################################
###################################################################



# Initialization
set.seed(second(Sys.time()));
param.train.ratio = 0.7;
param.validation.ratio = 0.1;
param.test.ratio = 0.2;

# Naive Mean Method
generate_dataset(d.train, im.train, param.train.ratio, param.validation.ratio, param.test.ratio);
predictions.naive = pred_algo_naive(d.train.train, d.train.test);
head(predictions.naive)
eval_prediction(predictions.naive[,2:31], d.train.test[,1:30]);
vis_img(im.train.train, d.train.train[,1:30], predictions.naive);
leanring_curve(seq(500,7000,500), d.train.train, d.train.test,pred_algo_naive);
#create_submission(predictions.naive,"submission_apr_6.csv");

# Naive Image Algorithm : NOT WORKING
generate_dataset(d.train, im.train, param.train.ratio, param.validation.ratio, param.test.ratio);
predictions.image = pred_algo_image(d.train.train, d.train.test);
eval_prediction(predictions.image[,2:ncol(predictions.image)], d.train.test);

# Linear Regression Algorithm
generate_dataset(d.train, im.train, param.train.ratio, param.validation.ratio, param.test.ratio);
predictions.lm = pred_algo_lm(d.train.train, d.train.test[,31:37]);
eval_prediction(predictions.lm[,2:ncol(predictions.lm)], d.train.test[1:30]);
vis_img(im.train.train, d.train.train[,1:31], predictions.lm);
leanring_curve(seq(500,7000,500), d.train.train, d.train.test,pred_algo_lm);
predictions.lm.submission = pred_algo_lm(d.train, feature.test);
create_submission(predictions.lm.submission,"submission_lr.csv");






###################################################################
######### Visualization Code ######################################
###################################################################








for (i in 1:nrow(im.test)) {
  im <- matrix(data=rev(im.test[i,]), nrow=96, ncol=96);
  image(1:96, 1:96, im, col=gray((0:255)/255));
  line = readline();
  cat(i,"\n")
  if (line == 'break') {
      break;
  }
}


# vis key points
im <- matrix(data=rev(im.test[i,]), nrow=96, ncol=96);
image(1:96, 1:96, im, col=gray((0:255)/255));
points(96-44.6,96-42.3, col="red")

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



###################################################################
######### Workspace ######################################
###################################################################

# Test LM
lm.models = lm(left_eye_center_x ~ centroid_x+centroid_y+ellipse_center_x+ellipse_center_y, data=d.train);
summary(lm.models);


# Create Image
im <- matrix(data=rev(im.train[51,]), nrow=96, ncol=96);
im = rotate(im, 90);



# biOps
img = imagedata(im);
plot(imagedata(img))
imageType(img)
y = imgCanny(img, sigma = 1.5)
plot(imagedata(y))

#image(1:96, 1:96, im, col=gray((0:255)/255));
im = rotate(im, 180);
img = Image(im);
colorMode(img)=Grayscale;
print(img);
display(img);



# brush
#flo = makeBrush(21, shape = 'disc', step = FALSE)^2;
#flo = flo/sum(flo);
#lenaflo = filter2(img, flo);
#display(lenaflo)
fhi = matrix(1, nc=3, nr=3);
fhi[2,2] = -8;
lenafhi = filter2(img, fhi)
display(lenafhi)
lenafhi[lenafhi>=0.3] = 1;
lenafhi[lenafhi< 0.3] = 0;
display(lenafhi)

# countour
con = contour(lenafhi);

#
nmask = thresh(img, 10, 10, 0.5);
display(img);



cal_diff<-function(x, y, img){
  l = c();
  x_s = ifelse(x-1<=0, 1, x-1 );
  x_e = ifelse(x+1>96, 96, x+1 );
  y_s = ifelse(y-1<=0, 1, y-1 );
  y_e = ifelse(y+1>0, 96, y+1 );
  
  for (i in x_s:x_e) {
    for (j in y_s:y_e) {
      l[length(l)+1] = abs(img[i][j]-img[x][y]);
    }
  }

  return(mean(l));
  
}


# vis key points
points(96-d.train$nose_tip_x[1],         96-d.train$nose_tip_y[1],         col="red");
points(96-d.train$left_eye_center_x[1],  96-d.train$left_eye_center_y[1],  col="blue");
points(96-d.train$right_eye_center_x[1], 96-d.train$right_eye_center_y[1], col="green");

