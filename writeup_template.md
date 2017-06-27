**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

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

The `extract_features()` function was used to extract features for all of the images at once.  Around line 108, I used HOG feature detection and combined that with spatial-binning (~line 100) and color-histograms (~line 103).  Once all three feature-types were concatenated, I had to normalize them using a StandardScaler (way down near line 631) so that they would work together instead of one feature-type dominating the classifications.

This figure shows an example image from the training-data, as well as its feature-extractions and what the features look like after normalization.

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
I could probably have gotten away with a smaller "histbin" value, but every time I change one of those values, I have to re-train the classifier which is very time-consuming. Since I was already running fast enough and with extremely high accuracy (typically 0.9986), I directed my attention onward.

Since I was starting with great parameters that I knew were good from playing around in the lessons ealier, my classifier was above 0.99 almost immediately, so the only additional optimizations I did were related to making it run faster without degrading accuracy.

I noticed that 'HSV' and 'HLS' performed very similar in my earlier experiments but they were both very good. I did not test the well-regarded 'YCrCb' color-space on my project (although I played with it earlier and got similar results to HLS), but that is another option that could have been interesting to explore if I needed to get even higher accuracy.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a Linear State Vector Classifier using 80% of the aforementioned 17,000+ example images, and used the remaining 20% for validation.  The sets are shuffled randomly using `train_test_split()` on every execution of the code, so I can see that the classifier is not being overfitted to one random selection of data.

The training is done around line 630 of `findVehicles.py` (but this only executes if the classifier is not loaded from file).

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Cars aren't expected to be in the sky or up in the trees (or if they are, it's not terribly relevant), so I imediately decided to only search the area of the video that included road.  This included the bottom of the image and not all of the left side.  In the area that was worth searching, I used two sizes: 128px and 64px, each with a 50% overlap.  The 64px windows will only find small cars, so they are only needed "far away" (ie: towards the top of the search area).  I intentionally offset the 64px just to have them search a slightly different area (since 64 is half of 128, the overlaps were almost exact prior to adding the offsset).

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

To additionally remove false-positives, I started remembering the heatmaps from the 10 most recent frames and removing detections that did not have at least 4 detected boxes in those 10 frames (keep in mind that some pixels will have more than one detection in the same frame, if overlapping windows found detections).

I experimented with other methods (such as recording 5 historical frames and only giving heat in pixels that were detected in all 5 frames) but the 
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The approach I took was to build the pipeline step-by-step and debug that it was behaving. Unfortunately, there were a few hiccups. The first issue was that when I implemented HOG sub-sampling, I expected a much more significant performance improvement (because of how much it was talked about).  In reality, my test video sped up, but processing the entire project-video actually took LONGER.  I spent quite a few hours trying to figure out what I was doing wrong before finally coming to the realization that I am likely doing it correctly and that the bulk of the time being spent just lies elsewhere.

A significant problem for me was that I was getting pretty good detections on the static images almost immediately, but would come up with a ton of false-positives on the videos.  The general detections made sense (other than the false-positives) so I didn't think anything was wrong with my pre-processing at first.  I looked for hours to find what could be wrong with my parameters or what could be different between the static images and the video images.  Eventually this was solved by outputting a static image and a similar frame from the video, AFTER color-processing. The results were significantly different.  This helped me realize that the video frames **should** be re-scaled from 0-225 to 0-1 as if they were jpgs.  Earlier attempts at this were something I thought were incorrect because my output video had turned solid black.  The trick is just that for the video-stream, I needed to make sure that my box-drawings were done to the original image (prior to scaling) whereas the static images didn't seem to mind being outputted after being scaled.

Static Image vs Video-Frame after color conversion (without jpg scaling of video frame)
<img src="https://raw.githubusercontent.com/SeanColombo/CarND-Vehicle-Detection/master/output_images/without_scaling.png" width="600">

Static Image vs Video-Frame after color conversion (with jpg scaling of video frame)
<img src="https://raw.githubusercontent.com/SeanColombo/CarND-Vehicle-Detection/master/output_images/with_scaling.png" width="600">

With this conversion fixed, the remainder of the pipeline performed as desired.
