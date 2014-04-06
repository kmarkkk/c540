install.packages('foreach')
install.packages('doSNOW')
library(foreach)
library(doSNOW)

# Load Data
data.dir="~/Academy/2014 SPRING/COMP 540/Term Project/Facial Expression/data/";
setwd(data.dir)
train.file <- paste0(data.dir, 'training.csv')
test.file  <- paste0(data.dir, 'test.csv')
d.train <- read.csv(train.file, stringsAsFactors=F)

# Pre-preoccessing
str(d.train);
head(d.train);
im.train <- d.train$Image;
d.train$Image <- NULL;
head(d.train);

#Re-formatting

im.train <- foreach(im = im.train, .combine=rbind) %do% {
  as.integer(unlist(strsplit(im, " ")))
}
str(im.train)


d.test  <- read.csv(test.file, stringsAsFactors=F)
im.test <- foreach(im = d.test$Image, .combine=rbind) %dopar% {
  as.integer(unlist(strsplit(im, " ")))
}
d.test$Image <- NULL
save(d.train, im.train, d.test, im.test, file='data.Rd')
load('data.Rd')
