import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
import time
import argparse
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
# NOTE: the next import is only valid 
# for scikit-learn version >= 0.18
# if you are using scikit-learn <= 0.17 then use this:
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split






# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features
    
# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features
    

# Define a function to extract features from a list of images
# This function gets features from bin_spatial(), color_hist(), and get_hog_features().
def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256),
                        orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        color_spaces = {
            'HSV': cv2.COLOR_RGB2HSV,
            'LUV': cv2.COLOR_RGB2LUV,
            'HLS': cv2.COLOR_RGB2HLS,
            'YUV': cv2.COLOR_RGB2YUV,
            'YCrCb': cv2.COLOR_RGB2YCrCb
        }
        if cspace in color_spaces:
            feature_image = cv2.cvtColor(image, color_spaces[cspace])
        else:
            feature_image = np.copy(image)

        # RUBRIC POINT:
        # - Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.

        # TODO: Make this function use the use_spatial_feat, use_hist_feat, and use_hog_feat configurations also (single_img_feat() already does this).
        
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        
        # RUBRIC POINT:
        # - Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)

        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features, hog_features)))
        # TODO: Use this instead if we wanted to switch back to HOG-only (doesn't have as high of accuracy though).
        #features.append(hog_features)

    # Return list of feature vectors
    return features
    

def single_img_features(feature_image, spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        use_spatial_feat=True, use_hist_feat=True, use_hog_feat=True,
                        hog_features_param=None):
    """
    Returns the features for a single image (or in our use-case, a window portion of a larger image).
    This function is very similar to extract_features()
    just for a single image rather than list of images
    
    If hog_features is provided, then it is used directly. If it is "None" then it will be calculated
    by calling get_hog_features(). This is used because the parent function is doing
    HOG sub-sampling for performance reasons, so it may already know the HOG features
    for this image patch.
    """
    #1) Define an empty list to receive features
    img_features = []

    #2) NOTE: The feature_image passed in must ALREADY be in the correct color-space!

    #3) Compute spatial features if flag is set
    if use_spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if use_hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if use_hog_feat == True:
        # It's possible we already have the features passed in. If not, then we calculate them.
        if hog_features_param is None:
            #print("ACTUALLY EXTRACTING HOG FEATURES FOR AN INDIVIDUAL IMAGE. THIS SHOULD NOT HAPPEN WHEN SUBSAMPLING.") # TODO: REMOVE?
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_channel = get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True)
                    #print("Normal method's hog-channel shape: ", hog_channel.shape)
                    hog_features.extend(hog_channel)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        else:
            hog_features = hog_features_param
        #8) Append features to list
        #print("HOG SUBSMP SHAPE: ",len(hog_features_param)," - HOG NORMAL SHAPE: ",len(hog_features), "  " + " - HOG DATA MISMATCH!! " if len(hog_features_param) != len(hog_features) else "")
        #print("HOG SUBSMP: ",hog_features_param[0:10])
        #print("HOG NORMAL: ",hog_features[0:10])
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)


# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
        
    # TODO: Could make it more robust by ensuring that START is low values and STOP is high values.

    # Compute the span of the region to be searched
    search_start = (x_start_stop[0], y_start_stop[0])
    search_stop = (x_start_stop[1], y_start_stop[1])
    x_overlap = (xy_window[0]*xy_overlap[0])
    y_overlap = (xy_window[1]*xy_overlap[1])

    # Compute the number of pixels per step in x/y. This is the
    # same as saying, what is the NON-overlapping part of the window.
    pixels_per_step_x = (xy_window[0] - x_overlap)
    pixels_per_step_y = (xy_window[1] - y_overlap)

    # Compute the number of windows in x/y
    search_width = x_start_stop[1] - x_start_stop[0]
    search_height = y_start_stop[1] - y_start_stop[0]
    num_windows_x = int((search_width - x_overlap) / pixels_per_step_x)
    num_windows_y = int((search_height - y_overlap) / pixels_per_step_y)
    
    # Good for debugging, if the windows don't render.
    # print("")
    # print("WINDOW-CREATION VALUES...")
    # print("Search start: ",search_start)
    # print("Search stop: ",search_stop)
    # print("Overlap: ",x_overlap,", ",y_overlap)
    # print("PIXELS PER STEP: ",pixels_per_step_x,", ",pixels_per_step_y)
    # print("Search area size: ",search_width,", ",search_height)
    # print("Num Windows X: ",num_windows_x)
    # print("Num Windows Y: ",num_windows_y)

    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    #     Note: you could vectorize this step, but in practice
    #     you'll be considering windows one by one with your
    #     classifier, so looping makes sense
    for win_num_x in range(num_windows_x):
        for win_num_y in range(num_windows_y):
            # Calculate each window position
            x = int(search_start[0] + (win_num_x * pixels_per_step_x))
            y = int(search_start[1] + (win_num_y * pixels_per_step_y))
            window = ( (x, y), (x + xy_window[0], y + xy_window[1]) )

            # Append window position to list
            window_list.append( window )

    # Return the list of windows
    return window_list


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # make a copy of the image
    draw_img = np.copy(img)
    # draw each bounding box on your image copy using cv2.rectangle()
    for box in bboxes:
        cv2.rectangle(draw_img, box[0], box[1], color, thick)
    # return the image copy with boxes drawn
    return draw_img


def search_windows(img, windows, clf, scaler,
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, use_spatial_feat=True, 
                    use_hist_feat=True, use_hog_feat=True,
                    hog_channels=None,
                    do_output=False, image_name=""):
    """
    Given an image, a list of windows to search in, and a trained classifier & scaler...
    this will search each window on the image and try to classify whether it contains a car.
    
    The result will be an image which has bounding-boxes drawn on areas that appear to be a car.
    
    "windows" is an array of 2-tuples where the first value of the tuple is the coordinates of the
    top-left corner of the window and the second value is the coordinates of the bottom-right corner
    of the window.
    
    If hog_channels is provided, it is assumed to be a 3-tuple of hog information (each item in the
    tuple is the hog features for one channel). If "None", then the HOG features will be calculated
    from scratch for each window.
    """
    
    # We can use HOG subsampling if the full-image HOG features are provided (per-channel)
    if hog_channels is not None:
        (hog1, hog2, hog3) = hog_channels
        # NOTE: The test_img is scaled to 64x64, so that's the only width/height dimensions we need.
        window_width = 64#test_img.shape[0]#window[1][0] - window[0][0]
        window_height = 64#test_img.shape[1]#window[1][1] - window[0][1]
        nblocks_per_window_x = (window_width // pix_per_cell) - cell_per_block + 1
        nblocks_per_window_y = (window_height // pix_per_cell) - cell_per_block + 1

    # RUBRIC POINT:
    # Implement a sliding-window technique and use your trained classifier to search for vehicles in images
    
    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    did_output = False
    for window in windows:
        #3) Extract the test window from original image... always ensure that the output
        #   is 64x64 to match training data (that's all the classifier can work with).
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        
        # Do HOG sub-sampling if available.
        if hog_channels is None:
            hog_sub_features = None
        else:
            # Extract HOG features for this patch
            xpos = window[0][0] // pix_per_cell
            ypos = window[0][1] // pix_per_cell
            # TODO: FIXME: This assumes we're using "ALL" channels. It shouldn't assume that even though that's all we're going to use.
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window_y, xpos:xpos+nblocks_per_window_x].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window_y, xpos:xpos+nblocks_per_window_x].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window_y, xpos:xpos+nblocks_per_window_x].ravel()
            #print("HOG SUBSAMPLED INDIVIDUAL CHANNEL: ",hog_feat1.shape)
            hog_sub_features = np.hstack((hog_feat1, hog_feat2, hog_feat3)).ravel()

        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img,
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, use_spatial_feat=use_spatial_feat, 
                            use_hist_feat=use_hist_feat, use_hog_feat=use_hog_feat,
                            hog_features_param=hog_sub_features)

        #5) Scale extracted features to be fed to classifier
        reshaped_features = np.array(features).reshape(1, -1).astype(np.float64)
        test_features = scaler.transform(reshaped_features)

        #6) Predict using your classifier
        prediction = clf.predict(test_features[0])

        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
            
            # This is really slow but was helpful.
            # if do_output and not did_output:
                #Debug the feature extraction and scaling for a single window and make sure it is parallel to what
                #we do for the training.
                # fig = plt.figure(figsize=(12,4))
                # plt.subplot(131)
                # plt.imshow(test_img)
                # plt.title('Original Image')
                # plt.subplot(132)
                # plt.plot(features)
                # plt.title('Raw Features')
                # plt.subplot(133)
                # plt.plot(test_features[0])
                # plt.title('Normalized Features')
                # fig.tight_layout()
                # plt.savefig(os.path.join(OUT_DIR, "x-normalized-vs-undistorted-singleimage-"+image_name+".png"))
                # plt.close()
                # did_output = True # prevents saving a file for EVERY window
            

    #8) Return windows for positive detections
    return on_windows

def process_image(image, do_output=False, image_name="", image_was_jpg=False):
    """
    Given an image (loaded from a file or a frame of a video), 
    process it to find the vehicles and draw bounding boxes around them.

    image:      the full-color image (eg: from cv2.imread()).
    do_output:  whether to output images of the various steps. Intended to be done doing
                for the static images but not for the videos (since there are a ton of frames).
    image_name: optional. Recommended when do_output is true. This will be used for debug
                and output-filenames related to this image.
    image_was_jpg: If true, then we assume the input image was a jpg (video frames are not jpgs) which
                   means that we need to scale the color to be 0-1 instead of 0-255 to match the
                   training-images that were loaded as pngs.
    """
    global recent_hot_windows

    image_name, image_extension = os.path.splitext(image_name)
    
    # Scale the colors from the range 0-255 to be 0-1 which matches training images.
    if image_was_jpg:
        image = image.astype(np.float32)/255
    box_color = (0,0,1.0) if image_was_jpg else (0,0,255)


    # Get the sliding-window boundaries that we'll search for cars.
    # We use just the bottom area of the images since we don't expect flying-cars to interfere with our driving ;)
    y_start = int(image.shape[0] * 0.55)
    x_start = int(image.shape[1] * 0.35) # ignore the far-left.. it's the shoulder of the road
    windows = slide_window(image, x_start_stop=[x_start, None], y_start_stop=[y_start, None], 
                           xy_window=(128, 128), xy_overlap=(0.5, 0.5))

    # Use multiple scales of windows... here we add a second scale which has much smaller windows, only processes
    # the top part of our area of interest (because that's the only place that cars are that small) and adds these
    # windows to our list
    y_start -= 16 # just to stagger it a bit from the bigger windows
    x_start = int(image.shape[1] * 0.45) # ignore the far-left.. it's the shoulder of the road
    y_stop = int(image.shape[0] * 0.80)
    windows.extend(slide_window(image, x_start_stop=[x_start, None], y_start_stop=[y_start, y_stop],
                           xy_window=(64, 64), xy_overlap=(0.5, 0.5)))
    #smaller_windows = slide_window(image, x_start_stop=[x_start, None], y_start_stop=[y_start, y_stop],
    #                       xy_window=(64, 64), xy_overlap=(0.5, 0.5))
    if do_output:
        window_image = draw_boxes(image, windows, color=box_color, thick=6)
        #window_image = draw_boxes(window_image, smaller_windows, color=(1.0,0,0), thick=4) # DEBUG: This was just used to render the smaller scaled windows
        plt.imsave(os.path.join(OUT_DIR, "010-all-windows-"+image_name+".png"), window_image)
        plt.close()
        
    # Extract the HOG features for the whole image here, then we will pass this into search_windows
    # which will sub-sample from this array to get the HOG features for each desired window.
    color_spaces = {
        'HSV': cv2.COLOR_RGB2HSV,
        'LUV': cv2.COLOR_RGB2LUV,
        'HLS': cv2.COLOR_RGB2HLS,
        'YUV': cv2.COLOR_RGB2YUV,
        'YCrCb': cv2.COLOR_RGB2YCrCb
    }
    if colorspace in color_spaces:
        converted_image = cv2.cvtColor(image, color_spaces[colorspace])
    else:
        converted_image = np.copy(image)
    ch1 = converted_image[:,:,0]
    ch2 = converted_image[:,:,1]
    ch3 = converted_image[:,:,2]
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
    
    # Do the sliding-window search across the image to find "hot" windows where it appears
    # that there is a car.
    hot_windows = search_windows(converted_image, windows, svc, X_scaler,
                        spatial_size=(spatial, spatial), hist_bins=histbin, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, use_spatial_feat=use_spatial_feat, 
                        use_hist_feat=use_hist_feat, use_hog_feat=use_hog_feat,
                        hog_channels=(hog1, hog2, hog3),
                        do_output=do_output, image_name=image_name)

    # Instead of drawing the bounding-boxes directly, we'll use a heatmap to find the best fits.
    hot_windows_instantaneous = draw_boxes(image, hot_windows, color=box_color, thick=6)
    if do_output:
        plt.imsave(os.path.join(OUT_DIR, "011-hot-windows-"+image_name+".png"), hot_windows_instantaneous)
        plt.close()
        
        plt.imsave(os.path.join(OUT_DIR, "010-boxes-"+image_name+".png"), window_image)
        plt.close()

    # TODO: HEATMAPS TO PREVENT FALSE POSITIVES AND COMBINE OVERLAPPING DETECTIONS.
    # == HEATMAPPING THE RECENT X FRAMES ==
    NUM_FRAMES_TO_REMEMBER = 5
    MIN_BOXES_NEEDED = 15 # remember: there are multiple (often overlapping) boxes per video-frame
    while( len(recent_hot_windows) >= NUM_FRAMES_TO_REMEMBER ):
        # Deletes the oldest set of hot windows
        del recent_hot_windows[0]
    recent_hot_windows.append( hot_windows ) # adds the new frame's hot windows
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    
    # Add heat to each box in box list - TODO: DO WE NEED TO ITERATE FOR THIS?
    # This method just summed them up and then we used a threshold.
    # for hot_wins in recent_hot_windows:
        # heat = add_heat(heat, hot_wins)
    # This method ensures that the pixel was in ALL five of the last frames.
    running_heatmap = add_heat(heat, recent_hot_windows[0])
    for hot_wins in recent_hot_windows:
        heat = np.zeros_like(image[:,:,0]).astype(np.float)
        heat = add_heat(heat, hot_wins)
        running_heatmap[heat == 0] = 0
    
    # Apply threshold to help remove false positives
    #heat = apply_threshold(heat, MIN_BOXES_NEEDED)
    heat = apply_threshold(running_heatmap, 2)
    
    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    hot_window_image = draw_labeled_bboxes(np.copy(image), labels, color=box_color)

    if do_output:
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(hot_windows_instantaneous) # instantaneous is the raw boxes
        plt.title('Car Positions')
        plt.subplot(122)
        # Render an individual heatmap rather than an averaged one
        heat = np.zeros_like(image[:,:,0]).astype(np.float)
        heat = add_heat(heat, hot_windows)
        plt.imshow(heat, cmap='hot')
        plt.title('Heat Map')
        fig.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "015-heatmap-"+image_name+".png"))
        plt.close()    

    # TODO: SWAP BACK TO USE THE BOUNDING BOXES FROM HEATMAP... RETURNING "instantaneous" IS ONLY TO HELP DEBUG FALSE-POSITIVE BOX MATCHES.
    return hot_window_image
    #return hot_windows_instantaneous

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels, color=(0,0,255)):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], color, 6)
    # Return the image
    return img


###############################################################
# MAIN EXECUTION BEGINS HERE!!!
    
script_start_time = time.time() # times the ENTIRE script
IN_DIR = "test_images"    
OUT_DIR = "output_images"
CAR_DIR = "../vehicles"
NOT_CAR_DIR = "../non-vehicles"
VIDEO_IN_DIR = "."
VIDEO_OUT_DIR = OUT_DIR # this is easier and the rubric doesn't seem to specify where we have to do it
USE_LOADED_CLASSIFIER = True
SAVE_CLASSIFIER = True # sometimes we might want to re-run but aren't sure we want to overwrite what we have

#### FEATURE PARAMETERS!! ####
spatial = 32
histbin = 64
colorspace = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 10 # 10 and 16 both do great. 10 is probably a little faster. Apparently the literature says it's useless to go above 9... but I saw visible degradation.
#orient = 16
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
use_spatial_feat = True # Spatial features on or off
use_hist_feat = True # Histogram features on or off
use_hog_feat = True # HOG features on or off
CLASSIFIER_FILENAME = "./classifier-ORIENT"+str(orient)+".pkl"
CAR_FEATURE_FILENAME = "./car_features-ORIENT"+str(orient)+".pkl"
NOTCAR_FEATURE_FILENAME = "./notcar_features-ORIENT"+str(orient)+".pkl"

# Ensure the output directory for images/videos exist so that we can write to them.
if not os.path.exists(VIDEO_OUT_DIR):
    os.makedirs(VIDEO_OUT_DIR)
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
    

# Some command-line options to let me choose what data to run the script on.
# By default it will process all types of media (static images, test video, project video).

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-nostatic', action='store_false', default=True, help='If provided, then the static images will NOT be processed')
parser.add_argument('-notest', action='store_false', default=True, help='If provided, then the short test video will NOT be processed')
parser.add_argument('-noproject', action='store_false', default=True, help='If provided, then the large project video will NOT be processed')
args = parser.parse_args()

# Our definitions of arg-parsing flip the values for us so the bools returned
# are whether we want to DO the action. This prevents having to think through
# double-negatives.
DO_IMAGES = args.nostatic
DO_TEST_VIDEO = args.notest
DO_PROJECT_VIDEO = args.noproject


# == LOADING TRAINING/TEST DATA AND TRAINING/EVALUATING THE CLASSIFEIR ==
# Since this can be rather time-consuming, we save the output to a Pickle file
# and load the Pickle-file on subsequent runs.

# Load the training data from the directories for each class of images.
print("Loading list of training images...")
cars = glob.glob(CAR_DIR+"/**/*.png", recursive=True)
notcars = glob.glob(NOT_CAR_DIR+"/**/*.png", recursive=True)
print("NUM CAR IMAGES FOUND FOR TRAINING: ",len(cars))
print("NUM NON-CARS FOUND FOR TRAINING: ",len(notcars))

print("")
if USE_LOADED_CLASSIFIER and os.path.isfile(CAR_FEATURE_FILENAME) and os.path.isfile(NOTCAR_FEATURE_FILENAME):
    print("Loading feature extractions from file...")
    t=time.time()
    car_features = joblib.load(CAR_FEATURE_FILENAME)
    notcar_features = joblib.load(NOTCAR_FEATURE_FILENAME)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to load feature-extractions from file.')
else:
    print("Extracting features...")
    t=time.time()
    car_features = extract_features(cars, cspace=colorspace, spatial_size=(spatial, spatial),
                            hist_bins=histbin, hist_range=(0, 256),
                            orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
    notcar_features = extract_features(notcars, cspace=colorspace, spatial_size=(spatial, spatial),
                            hist_bins=histbin, hist_range=(0, 256),
                            orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to extract features.')
    if SAVE_CLASSIFIER:
        # Saving extracted features since that's very time-consuming.
        print("Saving feature extractions (this take quite a while since it compresses the data)...")
        joblib.dump(car_features, CAR_FEATURE_FILENAME, compress=9)
        joblib.dump(notcar_features, NOTCAR_FEATURE_FILENAME, compress=9)

# RUBRIC POINT:
# - Don't forget to normalize your features
if len(car_features) > 0:
    print("Normalizing features...")
    t=time.time()
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to normalize features.')

    # Plot an example of raw and scaled features
    print("Plotting raw vs. normalized features...")
    car_index = np.random.randint(0, len(cars)) # randomly choose a car image for this example
    fig = plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.imshow(mpimg.imread(cars[car_index]))
    plt.title('Original Image')
    plt.subplot(132)
    plt.plot(X[car_index])
    plt.title('Raw Features')
    plt.subplot(133)
    plt.plot(scaled_X[car_index])
    plt.title('Normalized Features')
    fig.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "x-normalized-vs-undistorted.png"))
    plt.close()
    print("Done with plot.")
else: 
    print('ERROR: The function only returns empty feature vectors!')
    exit()

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
# RUBRIC POINT:
# - don't forget to randomize a selection for training and testing.
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

print("")
if USE_LOADED_CLASSIFIER and os.path.isfile(CLASSIFIER_FILENAME):
    print("Loading saved classifier from file...")
    svc = joblib.load(CLASSIFIER_FILENAME)
else:
    print('Using spatial binning of:',spatial,'and', histbin,'histogram bins')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    if SAVE_CLASSIFIER:
        # Since the classifier takes a while to train, we will save it to file.
        joblib.dump(svc, CLASSIFIER_FILENAME, compress=9)

# Check the score of the SVC - this is done outside of the saving/loading branches
# as a sanity-check in case something is wrong with our saved version of the
# classifier. If the classifier that is saved is bad, then we'll see that here.
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

# A global which will store a list of the most recent X collections of hot-windows, for use in heatmapping.
recent_hot_windows = []

# == STATIC IMAGES FOR TESTING OUR SOLUTION ==
# Process and save each file that exists in the input directory.
if DO_IMAGES:
    print("")
    print("Processing static images...")
    files = os.listdir(IN_DIR)
    for file_index in range(len(files)):
        fullFilePath = os.path.join(IN_DIR, files[file_index])

        # All of the image-processing is done in this call
        print("     Processing "+fullFilePath+"...")
        image = mpimg.imread(fullFilePath)
        
        # Uncomment the following line if you extracted training
        # data from .png images (scaled 0 to 1 by mpimg) - WHICH WE ARE
        # ...and the image you are searching is a .jpg (scaled 0 to 255)
        image_name, image_extension = os.path.splitext(files[file_index])
        image_extension = image_extension.lower()
        image_was_jpg = ((image_extension == ".jpg") or (image_extension == ".jpeg"))
        if(image_was_jpg):
            print("     NOTE: Training was done on .png's and we are evaluating a .jpg, so we are scaling the pixel values to match.")

        image = process_image(image, do_output=True, image_name=files[file_index], image_was_jpg=image_was_jpg)

        # Take the processed image and save it to the output directory.
        saveFile = os.path.join(OUT_DIR, files[file_index])
        plt.imsave(saveFile, image)
    print("Done processing static images.")
    
# Video processing
if DO_TEST_VIDEO:
    recent_hot_windows = []
    print("Processing short video file...")
    video_input_filename = os.path.join(VIDEO_IN_DIR, 'test_video.mp4')
    video_output_filename = os.path.join(VIDEO_OUT_DIR, 'test_video.mp4')
    clip1 = VideoFileClip(video_input_filename)
    output_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    print("Writing video file...")
    output_clip.write_videofile(video_output_filename, audio=False)

# print("Processing full project video file...")
if DO_PROJECT_VIDEO:
    recent_hot_windows = []
    video_input_filename = os.path.join(VIDEO_IN_DIR, 'project_video.mp4')
    video_output_filename = os.path.join(VIDEO_OUT_DIR, 'project_video.mp4')
    clip1 = VideoFileClip(video_input_filename)
    output_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    print("Writing video file...")
    output_clip.write_videofile(video_output_filename, audio=False)




script_end_time = time.time()
print("Entire script took ",round(script_end_time-script_start_time, 2),"seconds\a\a")



# To debug the way-too-many false-positives in the video-stream:
#   - Output the feature vector normalization for the video-stream for a few frames and compare to static images.
#   - Output the images that are examined in search_windows... for static images and for the video.



# Possible extras:
#   Could probably use the 'vis' parameter of get_hog_features() at some point to output the feature-vectors we detect.
#   Could use extra data from Udacity dataset for training (unlikely this will be needed): https://github.com/udacity/self-driving-car/tree/master/annotations
#       (the classifier has really high accuracy already, so that's probably not one of the bigger problems in the pipeline)
