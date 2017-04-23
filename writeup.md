##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car-not-car.png
[image2]: ./output_images/car-hog.png
[image3]: ./output_images/notcar-hog.png
[image4]: ./output_images/car-spatial.png
[image5]: ./output_images/car-hist.png
[image6]: ./output_images/notcar-hist.png
[image7]: ./output_images/test0-search-windows.png
[image8]: ./output_images/test1-search-windows.png
[image9]: ./output_images/test2-search-windows.png
[image10]: ./output_images/test3-search-windows.png
[image11]: ./output_images/test4-search-windows.png
[image12]: ./output_images/test5-search-windows.png
[image13]: ./output_images/test0-hot-windows.png
[image14]: ./output_images/test1-hot-windows.png
[image15]: ./output_images/test2-hot-windows.png
[image16]: ./output_images/test3-hot-windows.png
[image17]: ./output_images/test4-hot-windows.png
[image18]: ./output_images/test5-hot-windows.png
[image19]: ./output_images/test0-labels.png
[image20]: ./output_images/test1-labels.png
[image21]: ./output_images/test2-labels.png
[image22]: ./output_images/test3-labels.png
[image23]: ./output_images/test4-labels.png
[image24]: ./output_images/test5-labels.png
[video1]: ./project_video_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 36 through 53 of the file called `utils.py` and in lines 90 through 101 of the `single_img_features` method (in the same file, `utils.py`).

I started by reading in all the `vehicle` and `non-vehicle` images (code lines 130-131).  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces (`RGB`, `HSV` and `YCrCb`) and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=7`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` for `vehicle` and `non-vehicle`:


![alt text][image2]
![alt text][image3]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and the one with the best trade-off between accuracy and training time is what I chose for my model.

I ultimately landed on using the `YCrCb` color space and HOG parameters of `orientations=7`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` because they yeilded very good accuracy against the test set during training (0.9921) in a relatively short time period (10.76 seconds).Increasing the number of orientations (9, 18) and the number of histogram bins (to 64, for the color histogram features) didn't improve the accuracy by a significant margin, but took longer to train.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the HOG features descibed above, as well as spatial features and color histogram features, as follows:

	1. I extracted the `HOG` features that yielded the best accuracy against the test set on ALL 3 color channels.
	2. I extracted the `Spatial` features, with a spatial size of (32, 32).
	3. I extracted the `Color Histogram` features with 32 bins for each of the 3 color channels.
	4. I concatenated the 3 feature vectors and I then stacked both cars and not cars feature vectors in a single array X, corresponding to my training features (code line 145). This resulted in a data structure of shape (17760, 7284).
	5. Using `sklearn.preprocessing.StandardScaler()`, I normalized the feature vectors for training the classifier (code line 147).
	6. Then, I created the training labels by using `numpy.hstack()`, which assigns the label `1` for each `car` example and `0` for each `not car` example (line code 152).
	7. Using  `sklearn.model_selection.train_test_split`, I split up the data into randomized 80% training and 20% test sets (code lines 155-157). This resulted in 3552 test examples and 15984 training examples.
	8. Finally, I used `sklearn.svm.LinearSVC` to train a linear SVM classifier.

Here are some examples of Spatial binning and color histograms for a couple of images in the training set:

![alt text][image4]
![alt text][image5]
![alt text][image6]

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

For the purpose of this project and due to time considerations, I limited the search area to the bottom-right quarter of the image.

I used search windows of 2 scales: (120x96) and (280x224), with an overlap value of 0.9, as it resulted in the most accurate number of predictions across all frames.

Here are examples of test images depicting the search windows:

![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image13]
![alt text][image14]
![alt text][image15]
![alt text][image16]
![alt text][image17]
![alt text][image18]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps and bounding boxes:

![alt text][image19]
![alt text][image20]
![alt text][image21]
![alt text][image22]
![alt text][image23]
![alt text][image24]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

The greatest challenge I faced during this process was minimizing processing time for each detection operation. For this reason, I reduced the searching area to the bottom-right quarter of the images. This approach will fail if there are vehicles in front of the car, on the same lane, or on the left of the car (if the car travels on the center lane).

I would also track detected objects and smooth out the window over time using averaging algoithms for a cleaner detection experience.

I would also use HogSub-sampling Window Search as a more efficient method for doing the sliding window approach.
