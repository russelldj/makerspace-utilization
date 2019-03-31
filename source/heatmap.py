import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse
import pdb
import cv2
import glob
import json
from numpy.random import rand
from operator import itemgetter
from utils import openpose_loader
from PIL import Image


import KeypointCapture
import KeypointVisualization


"""
This script should take in the json results of running openpose (TODO link to openpose) and a refrence frame and and output a heatmap or scatter plot in floor space.
"""
JSON_FOLDER="data/Makerspace_mentors/jsons"
REF_IMAGE_FNAME = "data/Makerspace_mentors/ref_img.jpeg"
REF_VIDEO_FNAME = "data/Makerspace_mentors/ref_video.avi"
OVERHEAD_IMAGE_FNAME = "data/Makerspace_mentors/Makerspace.PNG"
EXAMPLE_NPY_HEATMAP = "results/unnormalized_heatmap0_100.npy"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-folder", help="The path to the folder of openpose jsons", default=JSON_FOLDER)
    parser.add_argument("--refrence-image", help="An image of the space", default=REF_IMAGE_FNAME)
    parser.add_argument("--overhead-image", help="The floorplan of your space", default=OVERHEAD_IMAGE_FNAME)
    parser.add_argument("--choose-homography-points", action="store_true", default=False, help="Select new points to compute the homography")
    parser.add_argument("--visualize-from-file", action="store_true", default=False, help="Visualize from a heatmap stored in a file")
    args = parser.parse_args()
    return args

def visualize(foldername=""):
    capture = KeypointCapture.Read2DJsonPath(foldername, "", "")
    visualizer = KeypointVisualization.KeypointVisualization()
    visualizer.WritePointsToVideo(capture, "first_makerspace_vid.avi", REF_VIDEO_FNAME)

def visualize_from_npy(filename, ref_img_fname=OVERHEAD_IMAGE_FNAME, colormap="inferno", heatmap_intensity=3.0, use_floormap=True):
    data = np.load(filename)
    data = data[...,2] # only one channel is visualized
    data /= np.amax(data) # normalize
    cmap = matplotlib.cm.ScalarMappable(cmap=colormap) # an object which turns scalars to colors
    #cmap.set_array(data)
    #cmap.autoscale()# norm the data from [0,1]
    cmapped = cmap.to_rgba(data)
    ref_im = cv2.imread(ref_img_fname) # read in the visualization image

    #cv2.imshow("", cv2.cvtColor(cmapped[...,:3].astype(np.uint8), cv2.COLOR_BGR2RGB))
    vis_heatmap = (cmapped[...,:3] * 255).astype(np.uint8)
    vis_heatmap = cv2.cvtColor(vis_heatmap, cv2.COLOR_BGR2RGB)
    if use_floormap: # visualize on the overhead view of the space
        vis_im = ref_im.astype(np.uint16) + (vis_heatmap*heatmap_intensity).astype(np.uint16)
    else:
        vis_im = vis_heatmap 
    vis_im = (vis_im / (np.amax(vis_im) / 255.0)).astype(np.uint8) # normalize to [0, 255]
    cv2.imshow("", vis_im)
    cv2.imwrite("colormapped.png", vis_im) 
    cv2.waitKey(2000)
    
def compute_homography(image_fname=REF_IMAGE_FNAME, overhead_fname=OVERHEAD_IMAGE_FNAME, use_example_homography=True):
    """
    This interfaces with the user to get the relation between the room and the image

    args
    ---------- 
    image_file : str
        the filepath to the refrence image of the scene
    use_example_points : bool
        Whether to use pre-computed corespondences for the test_video_2018-12-17
    
    returns
    ----------  
    homograpy : np.array
        the matrix representing the transformation from image-space to floor-space
    """
    if use_example_homography:
        #homography =np.array([[-7.37894113e-01, -4.29057777e-01,  2.12304130e+02],
        #                      [ 5.98959797e-02, -3.74192101e+00,  1.25389015e+03],
        #[ 2.07156462e-04, -4.45730030e-03,  1.00000000e+00]]) #This is for test
        #homography = np.array([[-5.48438309e-01, -4.07862594e-01,  1.25089007e+02],
        #                       [-9.07037579e-02, -3.80164086e+00,  1.16445525e+03],
        #                       [ 5.55292717e-04, -5.11058562e-03,  1.00000000e+00]])# this is hte one for the mentors
        #homography = np.array([[ 8.15519553e-01, -2.98981347e-02,  1.15961249e+01],
        #                       [-1.75840535e-02,  1.83333211e+00,  5.62610817e+00],
        #                       [-3.19357093e-05, -3.98188137e-05, 1.00000000e+00]])

        #homography = np.array([[-7.52717829e-01, -5.41937044e-01,  1.80284909e+02],           
        #                     [-3.64769537e-01, -5.09755179e+00,  1.63437459e+03],                    
        #                     [ 8.44335295e-05, -4.62214990e-03,  1.00000000e+00]]) # new floor plan

        homography = np.array([[-1.83275276e+00, -1.52041828e-01,  6.00408440e+01],
                       [-1.02265870e+00, -8.82312556e+00,  2.94339777e+03],
                       [ 1.66278357e-05, -6.63657266e-03,  1.00000000e+00]]) # 8 points

        
        reference_image = Image.open(image_fname)
        overhead_image = Image.open(overhead_fname)
        test_warp(homography, np.array(reference_image), np.array(overhead_image).shape)

        return homography
    # todo visualize both the image and the blank canvas
    f, (ax1, ax2) = plt.subplots(1, 2)
    # note that matplotlib only accepts PIL images natively
    reference_image = Image.open(image_fname)
    overhead_image = Image.open(overhead_fname)
    ax1.imshow(reference_image)
    ax2.imshow(overhead_image)
    plt.title("click 16 coresponding points in each image, by starting with the left one and alternating")
    # get the clicked points
    points = plt.ginput(16, timeout=-1, show_clicks=True)

    #close all the plots
    plt.close()
    source_pts = np.asarray(itemgetter(0, 2, 4, 6, 8, 10, 12, 14)(points))
    dest_pts   = np.asarray(itemgetter(1, 3, 5, 7, 9, 11, 13, 15)(points))
    raw_input("source points {}, dest points {} ".format(source_pts, dest_pts))
    homography, status = cv2.findHomography(source_pts, dest_pts)
    raw_input("homography {} ".format(homography))

    test_warp(homography, np.array(reference_image), np.array(overhead_image).shape)
    return homography


def load_jsons(json_folder):
    """
    reads a folder of openpose output files into a list

    args
    ---------- 
    json_folder : str
        The path, with or without trailihg slash, that points to the files

    returns 
    ---------- 
    points_list : list[dict()]
        a list where each entry represents a frame
    """
    if json_folder[-1] != "/":
        json_folder += "/"
    files = sorted(glob.glob("{}*".format(json_folder)))
    frames = list()
    for file_ in files:
        with open(file_, 'r') as json_file:
            json_data = json.loads(json_file.read())
            simplified_people = openpose_loader.parse_json(json_data)
            frames.append(simplified_people)
    return frames


def gaussian(x_mean, y_mean, x_size, y_size, sigma=10 ):
    """
    x_mean, y_mean are the centers of the gaussian
    x_size, y_size are the sizes of space this is being plotted on
    sigma is the standard diviation of the gaussian

    Taken from https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-79.php
    """
    x, y = np.meshgrid(np.linspace(0, x_size, x_size), np.linspace(0, y_size, y_size))
    x -= x_mean
    y -= y_mean
    d = np.sqrt(x*x+y*y)
    mu = 0
    g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    return g

def test_warp(homography, input_im, output_shape):
    im_out = cv2.warpPerspective(input_im, homography, (output_shape[1], output_shape[0]))
    cv2.imshow("", im_out)
    cv2.imwrite("warped.png", im_out)
    cv2.waitKey(1)
    cv2.destroyAllWindows()

def select_floor_keypoint(person):
    """
    take a person object and returns the location in image space of the selected keypoint
    """
    #min(person, key=person.)
    return max(person.items(), key=lambda x: x[1][1])[1] # note that this is image space coordinates
    
def plot_points(frames, homography, ref_image_fname=REF_IMAGE_FNAME, start_ind=0, stop_ind=100, overhead_image_fname=OVERHEAD_IMAGE_FNAME, use_video=True, ref_video_fname=REF_VIDEO_FNAME, colormap="Blues"):
    # load all of the data and open the subplot
    f, (ax1, ax2) = plt.subplots(1, 2)
    if use_video:
        video = cv2.VideoCapture(ref_video_fname)
        _, ref_im = video.read()
        ref_im = cv2.cvtColor(ref_im, cv2.COLOR_BGR2RGB)
    else:
        ref_im = Image.open(ref_image_fname)
    overhead_im = cv2.imread(overhead_image_fname) # decrease the brightness to allow for the heatmap
    ax1.imshow(ref_im)

    # initialize some accumulators
    ref_heatmap = np.zeros_like(ref_im, dtype=np.float64)[:,:,0] # create an accumulator for the heatmap that is one layer but the same shape as the image
    overhead_heatmap = np.zeros_like(overhead_im, dtype=np.float64) # create an accumulator for the heatmap
    ref_heatmap_y, ref_heatmap_x = ref_heatmap.shape
    overhead_heatmap_y, overhead_heatmap_x, _ = overhead_heatmap.shape
    print("the number of frames is {}".format(len(frames)))
    
    for frame in frames[start_ind:stop_ind]:
        #_, ref_im = video.read()
        #ref_im = cv2.cvtColor(ref_im, cv2.COLOR_BGR2RGB)
        for person in frame:
            #HACK selecting the right heel as the point we care about, this should be made more inteligent
            H = person["RHeel"]
            H = select_floor_keypoint(person)
            if H.conf == 0: # this means it wasn't actually detected so we shouldn't plot it
                continue # go to the next itteration of the loop
            ax1.scatter([H.x], [H.y]) # plot on the image just for visualization

            ref_heatmap += gaussian(H.x, H.y, ref_heatmap_x, ref_heatmap_y) # increment the heatmap

            # convert to homogenous coordinates
            homogenous = np.asarray([[H.x], [H.y], [1.0]]) # add the last coordinate
            transformed = np.dot(homography, homogenous)
            transformed /= transformed[2] # normalize

            gaussian_addition = gaussian(transformed[0], transformed[1], overhead_heatmap_x, overhead_heatmap_y)

            #overhead_heatmap[:,:,0] = overhead_heatmap[:,:,0] + gaussian_addition # increment the heatmap
            #overhead_heatmap[:,:,1] = overhead_heatmap[:,:,1] + gaussian_addition # increment the heatmap
            overhead_heatmap[:,:,2] = overhead_heatmap[:,:,2] + gaussian_addition # increment the heatmap

    np.save("results/unnormalized_heatmap{}_{}.npy".format(start_ind, stop_ind), overhead_heatmap)
    overhead_heatmap /= np.amax(overhead_heatmap) / 128.0 # normalize it to [1, 128] note that this counts recent points more heavily
    print("The max of the heatmap is {}".format(np.amax(overhead_heatmap)))
    vis = overhead_im / 2.0 + overhead_heatmap.astype(np.uint8)
    cv2.imwrite("visualization{}_{}.png".format(start_ind, stop_ind), vis)
    ax2.imshow(vis.astype(np.uint8)) # add the heatmap, must be unit8 or it will think it's [0,1]
    cv2.imshow("", vis.astype(np.uint8))
    plt.pause(0.005)
    ax1.set_title("foot locations overlayed on first image")
    ax2.set_title("foot locations in the room space")
    #plt.show()
    plt.close()
    
def main():
    # there are going to be a few steps
    ## load the jsons into a usable format
    ## find the refrence image
    ## mark the corespondences between the refrence image and the canvas
    ## visualize the results 
    args = parse_args()
    #visualize(args.json_folder)
    if args.visualize_from_file:
        visualize_from_npy(EXAMPLE_NPY_HEATMAP)
    else:
        homography = compute_homography(args.refrence_image, args.overhead_image, not args.choose_homography_points)
        print(homography)
        #homography = np.zeros((3,3))
        frames = load_jsons(args.json_folder)
        STEP_SIZE = 100
        for i in range(100):
            plot_points(frames, homography, args.refrence_image, i * STEP_SIZE, (i+1) * STEP_SIZE)

            
if __name__ == "__main__":
    parse_args()
    main()
