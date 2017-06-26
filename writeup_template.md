**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one:

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images and outputting the sizes of the classes to see that they were significant:

```
NUM CAR IMAGES FOUND FOR TRAINING:  8792
NUM NON-CARS FOUND FOR TRAINING:  8968
```

Initial training was very time-consuming so I had the large feature-extraction from the 17,000+ test images, and the training of the classifier all get stored in huge Pickle files.

#### 2. Explain how you settled on your final choice of HOG parameters.

Since the datasets were of good size, I started into training the classifier & optimizing it with some of the better parameter customizations I had tried in the lessons.  Initially I used 16 orientations since that gave great results & didn't have too much time-cost on smaller datasets, but when processing the video, these extra orientations were not providing significant return on investment so I eventually decreased that back to 9 which the literature on HOGs suggests is the point of diminishing returns.  I did this experiment of decreasing orientations when my entire pipeline was set up, so I was able to see that the change in results was imperceptible above 10, but that 10 actually did noticably better than 9.

Some winning configuration parameters for me were:
```
spatial = 32
histbin = 64
colorspace = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 10
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
use_spatial_feat = True # Spatial features on or off
use_hist_feat = True # Histogram features on or off
use_hog_feat = True # HOG features on or off
```
I could probably have gotten away with a smaller "histbin" value, but every time I change one of those values, I have to re-train the classifier which is very time-consuming.

Since I was starting with great parameters that I learned were good from playing around in the lessons ealier, my classifier was above 0.99 almost immediately and the only tweaking I did was related to making it run faster.

I noticed that 'HSV' and 'HLS' performed very similar in my earlier experiments but they were both very good.  Since I was above 0.99+ for most of my experiments (and changing these parameters involves significant time investment), I never tested the well-regarded 'YCrCb' color-space.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a Linear State Vector Classifier using 80% of the aforementioned 17,000+ example images, and used the remaining 20% for validation.  The sets are shuffled randomly using `train_test_split()` on every execution of the code, so I can see that the classifier is not being overfitted to one random selection of data.

The training is done around line 630 of `findVehicles.py` (but this only executes if the classifier is not loaded from file).

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Cars aren't expected to be in the sky or up in the trees (or if they are, it's not terribly relevant), so I imediately decided to only search the area of the video that included road.  This included the bottom of the image and not all of the left side.  In the area that was worth searching, I used two sizes: 128px and 64px, each with a 50% overlap.  The 64px windows will only find small cars, so they are only needed "far away" (ie: towards the top of the search area).  I intentionally offset the 64px.

Here, the small windows are rendered in red and the large windows in blue.  Keep in mind that these windows are overlapping each other by 50% in each direction, so the "squares" are 1/4 the dimension of the each window.

<img src="https://raw.githubusercontent.com/SeanColombo/CarND-Vehicle-Detection/master/output_images/x009-all-windows-two-sizes.png" width="600">

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?



TODO: YOU ARE HERE IN THE WRITEUP... BELOW IS JUST TEMPLATE/OUTLINE! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

