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

I used HOG feature detection and combined that with spatial-binning and color-histograms.  Once all three feature-types were concatenated, I had to normalize them so that they would work together instead of one feature-type dominating the classifications.

<img src="https://raw.githubusercontent.com/SeanColombo/CarND-Vehicle-Detection/master/output_images/x-normalized-vs-undistorted-zTHREE-FEATURE-TYPES.png" width="800">

#### 2. Explain how you settled on your final choice of HOG parameters.

Since the datasets were of good size, I started into training the classifier & optimizing it with some of the better parameter customizations I had tried in the lessons.  Initially I used 16 orientations since that gave great results & didn't have too much time-cost on smaller datasets, but when processing the video, these extra orientations were not providing significant return on investment so I eventually decreased that back to 9 which the literature on HOGs suggests is the point of diminishing returns.  I did this experiment of decreasing orientations when my entire pipeline was set up, so I was able to see that the change in results was imperceptible above 10, but that 10 actually did noticably better than 9.

Since I used spatial-binning and color-histograms in addition to HOG features, I tweaked settings for those also.

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

The sliding-window search was classifying each window it evaluated as being a "car" or "not car".  When a window was encountered that was a "car", that bounding box was tracked.  Initially this was just displayed on the data to test how we were doing.

<img src="https://raw.githubusercontent.com/SeanColombo/CarND-Vehicle-Detection/master/output_images/011-hot-windows-test1%20-%20Copy.png" width="600">

Interestingly, this did quite well on static images and was attrocious on the videos (way too many false positives).

The only way that I could find to mitigate these false-positives (which were a bit baffling since my classifier was so accurate in general) was to use heatmaps (more on that below).

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

To filter out false positives (of which there were many) and combine overlapping-detections, I implemented heatmaps around line 440 of `findVehicles.py`.

Examples of an image with bounding-boxes drawn (and the resulting heatmap) is here:
<img src="https://raw.githubusercontent.com/SeanColombo/CarND-Vehicle-Detection/master/output_images/heatmap_example.png" width="600">

<img src="https://raw.githubusercontent.com/SeanColombo/CarND-Vehicle-Detection/master/output_images/heatmap_example_2.png" width="600">

With the heatmaps in hand, I used `scipy.ndimage.measurements.label()` to identify blobs in the heatmap and constructed bounding-boxes around the blobs.

To additionally remove false-positives, I started remembering the heatmaps from the 5 most recent frames and removing detections that were not significant across several frames.


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The approach I took was to build the pipeline step-by-step and debug that it was behaving. Unfortunately, there were a few hiccups. The first issue was that when I implemented HOG sub-sampling, I expected a much more significant performance improvement (because of how much it was talked about).  In reality, my test video sped up, but processing the entire project-video actually took LONGER.  I spent quite a few hours trying to figure out what I was doing wrong before finally coming to the realization that I am likely doing it correctly and that the bulk of the time being spent just lies elsewhere.

Additionally, my pipeline performs greate on the static images but has an insane amount of false-positives in the video streams and I can't find any reason for that.
