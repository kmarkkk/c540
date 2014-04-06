# Load enviornment
data.dir="~/Academy/2014 SPRING/COMP 540/Term Project/Facial Expression/data/";
setwd(data.dir);
load('data.Rd');

# Visualize the image
im <- matrix(data=rev(im.train[1,]), nrow=96, ncol=96);
image(1:96, 1:96, im, col=gray((0:255)/255));
