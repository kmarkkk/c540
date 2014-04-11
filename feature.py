import numpy as np
import cv2
import csv
import os.path
import matplotlib.pyplot as plt
import sys

# Global Params
dir = 'C:\\Users\\Artie Shen\\Documents\\Academy\\2014 SPRING\\COMP 540\\Term Project\\Facial Expression\\data\\'
test_in = 'test_img.csv'
test_out = 'test_feature.csv'
train_in = 'train_img.csv'
train_in_subset = 'train_img_subset.csv'
train_out = 'train_feature.csv'
train_out_subset = 'train_feature_subset.csv'
BINARY_THREASHOLD = 1.0 # number of sigmas
FEATURE_SCHEMA = ['centroid_x', 'centroid_y', 'contour_area', 'countour_perimeter', 'ellipse_center_x', 'ellipse_center_y', 'eccentricity']

# Function that gets the centroid from the contour
def get_centroid(M):
	cx = int(M['m10']/M['m00'])
	cy = int(M['m01']/M['m00'])
	return (cx,cy)

	
# Function that set the threshold for an image
def get_threshold(image, threshold):
	image_mean = np.mean(image)	
	image_std = np.std(image)
	absolute_threshold = image_mean + threshold * image_std;
	if absolute_threshold > 127:
		absolute_threshold = 127
	return absolute_threshold
	
# Function that pre-processes the image
def pre_processing(image):
	image = np.array(image, dtype='uint8')
	for i in range(0,len(image)):
		image[0][i] = 0
		image[len(image)-1][i] = 0
		image[i][len(image)-1] = 0
		image[i][0] = 0
	
	return image
	
# Function that draws an ellipse on top of an image
def draw_ellipse(image, ellipse, ctr):
	cv2.ellipse(image,ellipse,(0,255,0),2)
	#cv2.imshow("img", image/255) 
	#cv2.waitKey(0) 	
	cv2.imwrite(dir+'python_output\\'+'img'+ str(ctr)+'e.png',image)
	

# Function that finds the edges of an image	
def find_edge(image):

	# Use Simple Threshold
	#absolute_threshold = get_threshold(image, 1.5)
	#ret, edge = cv2.threshold(image, absolute_threshold, 255, cv2.THRESH_BINARY) 
	#edge = np.array(edge,dtype='uint8')
	
	# Use Adaptive Threshold
	edge = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

	# Use Canny Edge Detection Algorithm
	#edge = cv2.Canny(image,100,200)
	
	
	edge = np.array(edge,dtype='uint8')
	return edge
	
# Function that return a list of contours	
def find_contour(image):
	# Use findContours() function
	contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	
	# Use HoughCircles() function
	#contours = cv2.HoughCircles(image, cv2.cv.CV_HOUGH_GRADIENT,minDist=1, dp=1,minRadius=1)#, minRadius=30,maxRadius=100)
	
	
	#print len(contours)
	return contours

	
# Function that processes the image and return a list of features
def process_image(image, threshold, ctr):
	# Find Countour
	image = pre_processing(image)
	edge = find_edge(image)
	contours = find_contour(edge)

	
	# Find Max Countour
	max_area = 0
	max_count = None
	max_moment = None
	for cont in contours:
		M = cv2.moments(cont)
		#print cv2.contourArea(cont)
		if max_area < cv2.contourArea(cont):
			max_area = cv2.contourArea(cont)
			max_count = cont
			max_moment = M

	# Debug		
	# print "image_mean = ", str(np.mean(image)), "\n"
	#cv2.imshow("img", binary/255) 
	#cv2.waitKey(0)	
	#cv2.drawContours(image,contours,-1,(0,0,255),2) 
	#cv2.imshow("img", image) 
	#cv2.waitKey(0)
	#cv2.imwrite(dir+'python_output\\'+'img'+ str(ctr)+'b.png',edge)

	
	#print "absolute_threshold = ", str(absolute_threshold), "\n"
	if (max_count == None or max_moment == None):
		print "contours=", str(contours), "max_area", str(max_area), "max_count = ", str(max_count), "max_moment = ", str(max_moment), "absolute_threshold = ", str(absolute_threshold),"\n"
		#cv2.drawContours(image,contours,-1,(0,0,255),2) 
		#cv2.imshow("img", binary/255) 
		#cv2.waitKey(0)
		return None
	
	# Contour Features
	(centroid_x, centroid_y) = get_centroid(max_moment);
	contour_area = cv2.contourArea(max_count)
	countour_perimeter = cv2.arcLength(max_count,True)
			
	# Ellipse Features
	(center, axes, orientation) = cv2.fitEllipse(max_count)
	minoraxis_length = min(axes)
	majoraxis_length = max(axes)
	eccentricity = np.sqrt(1-(minoraxis_length/majoraxis_length)**2)
	
	# Debug
	#draw_ellipse(image, (center, axes, orientation), ctr)
	
	return [centroid_x, centroid_y, contour_area, countour_perimeter, center[0], center[1], eccentricity]
	
	
	
# Top level function that reads a csv image file and write the features to the input file
def extract_cv_features(image_file, output_file, schema, threshold):
	# check if file exists
	if not os.path.isfile(image_file):
		print "extract_cv_features(): image_file = ", image_file, " is not a file\n" 
		return
	if not os.path.isfile(output_file):
		print "extract_cv_features(): output_file = ", output_file, " is not a file\n" 
		return
	
	# load image from the input csv file	
	print "extract_cv_features() : loading ", image_file, "\n"
	numpy_arr = np.loadtxt(open(image_file,"rb"),delimiter=",",skiprows=1)
	print "extract_cv_features() : finished loading ", image_file, ", received", str(len(numpy_arr)) ," images.\n"
	
	# process each image
	with open(output_file, 'wb') as output_csv:
		writer = csv.writer(output_csv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		writer.writerow(schema)
		for i in range(0,len(numpy_arr)):
			if i % 100 == 0:
				print "finished ", str(i), "/", str(len(numpy_arr)), "\n"
			
			# debug
			#print "Processing ",str(i),"th image \n"
			#plt.hist(numpy_arr[i])
			#plt.show()
				
			image = np.reshape(numpy_arr[i], (96,96))
			row = process_image(image, threshold, i)
			if row == None:
				print "extract_cv_features(): error with ", str(i), "\n"
				continue
			writer.writerow(row)




			
# Save Console output to file
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
f = open('out.txt', 'w')
original = sys.stdout
sys.stdout = Tee(sys.stdout, f)


#use the original
#sys.stdout = original
#print "This won't appear on file"  # Only on stdout
#extract_cv_features(dir+train_in, dir+train_out, FEATURE_SCHEMA, BINARY_THREASHOLD);
extract_cv_features(dir+test_in, dir+test_out, FEATURE_SCHEMA, BINARY_THREASHOLD);
f.close()

	
	





# with open(dir+'test_feature.csv', 'wb') as csvfile:	
	# spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	# spamwriter.writerow(['row1','row2','row3'])

	
# Load Image

#numpy_arr = np.loadtxt(open(dir+'test_img.csv',"rb"),delimiter=",",skiprows=1);
#first_img = np.reshape(numpy_arr[0], (96,96))
#np.save('test.txt', first_img)
#first_img = np.load(dir+'test.npy')
#cv2.namedWindow('image', 1)
#cv2.imshow('image',first_img/255)
#cv2.waitKey(0);

# Find Countour
#first_img = np.array(first_img,dtype='float32')
# for i in range(0,len(first_img)):
	# first_img[0][i] = 255
	# first_img[len(first_img)-1][i] = 255
	# first_img[i][len(first_img)-1] = 255
	# first_img[i][0] = 255
#ret, binary = cv2.threshold(first_img,100,255,cv2.THRESH_BINARY)  
#binary = np.array(binary,dtype='uint8')
#contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# cv2.imshow('image',first_img/255)
# cv2.waitKey(0);
# cv2.drawContours(first_img,contours,-1,(255,255,255),3) 
# cv2.imshow("img", first_img) 
# cv2.waitKey(0) 
# max_area = 0
# max_count = None
# for cont in contours:
	# M = cv2.moments(cont)
	# if max_area < cv2.contourArea(cont):
		# max_area = cv2.contourArea(cont)
		# max_count = cont

# print [max_area, "\n"]
# print get_centroid(cv2.moments(max_count))


# Ellipse
# ellipse = cv2.fitEllipse(max_count)
# cv2.ellipse(first_img,ellipse,(0,255,0),2)

#print ellipse
#cv2.drawContours(first_img,[max_count],-1,(0,255,0),3) 
# cv2.imshow("img", first_img/255) 
# cv2.waitKey(0) 

# Circle
#center, radius = cv2.minEnclosingCircle(max_count)
#cv2.circle(first_img, center, radius, ); 


#img = cv2.imread('test.jpg', cv2.IMREAD_COLOR)
#print img
#cv2.imshow('image',img)
#k = cv2.waitKey(0)
#if k == 27:         # wait for ESC key to exit
#    cv2.destroyAllWindows()
#elif k == ord('s'): # wait for 's' key to save and exit
#    cv2.imwrite('messigray.png',img)
#    cv2.destroyAllWindows()