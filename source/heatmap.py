import numpy as np
import matplotlib.pyplot as plt
import argparse
import pdb
import cv2
import glob
import json
from numpy.random import rand
from operator import itemgetter
from utils import openpose_loader
from PIL import Image


"""
This script should take in the json results of running openpose (TODO link to openpose) and a refrence frame and and output a heatmap or scatter plot in floor space.
"""

REF_IMAGE_FNAME = "data/DOWNSAMPLED_test_video_2018-12-17/room.jpg"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-folder")
    parser.add_argument("--refrence-image")
    args = parser.parse_args()
    return args

def compute_homography(image_fname=REF_IMAGE_FNAME, use_example_points=True):
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
    # todo visualize both the image and the blank canvas
    f, (ax1, ax2) = plt.subplots(1, 2)
    # note that matplotlib only accepts PIL images natively
    reference_image = Image.open(image_fname)
    ax1.imshow(reference_image)
    plt.title("click 4 coresponding points in each image, by starting with the left one and alternating")
    # get the clicked points
    #HACK TEMP
    if use_example_points:
        points = [(176.05693572002065, 1052.2532612289237), (0.03076559769517817, 0.00971235068656276), (850.3647806168233, 717.0943324044772), (0.016218769187172688, 0.9716444014438284), (1903.721414065083, 776.9441411231285), (0.9638407405658151, 0.9288505191083987), (1907.711401312993, 1020.333363245643), (0.8744816511594959, 0.4283481561418524)]
    else:
        points = plt.ginput(8, timeout=-1, show_clicks=True)

    #close all the plots
    plt.close()
    #pdb.set_trace()
    source_pts = np.asarray(itemgetter(0, 2, 4, 6)(points))
    dest_pts   = np.asarray(itemgetter(1, 3, 5, 7)(points))
    homography, status = cv2.findHomography(source_pts, dest_pts)
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
    files = glob.glob("{}*".format(json_folder))
    frames = list()
    for file_ in files:
        with open(file_, 'r') as json_file:
            json_data = json.loads(json_file.read())
            simplified_people = openpose_loader.parse_json(json_data)
            frames.append(simplified_people)
    return frames

def plot_points(frames, homography, ref_image_fname=REF_IMAGE_FNAME):
    f, (ax1, ax2) = plt.subplots(1, 2)
    im = Image.open(ref_image_fname)
    ax1.imshow(im)
    for frame in frames:
        for person in frame:
            #HACK selecting the right heel as the point we care about, this should be made more inteligent
            H = person["RHeel"]
            if H.conf == 0: # this means it wasn't actually detected so we shouldn't plot it
                continue # go to the next itteration of the loop
            ax1.scatter([H.x], [H.y]) # plot on the image just for visualization
            # convert to homogenous coordinates
            homogenous = np.asarray([[H.x], [H.y], [1.0]])# check that it's really x, y
            # this is a hack, I'm still unsure why the -1 needs to be there to get expected results
            transformed = -1.0 * np.dot(homography, homogenous)
            #plot the transformed ones
            ax2.scatter([transformed[0]], [transformed[1]])
    ax1.set_title("foot locations overlayed on first image")
    ax2.set_title("foot locations in the room space")
    plt.show()
    plt.waitforbuttonpress()

def main():
    # there are going to be a few steps
    ## load the jsons into a usable format
    ## find the refrence image
    ## mark the corespondences between the refrence image and the canvas
    ## visualize the results 
    args = parse_args()
    homography = compute_homography()
    #homography = np.zeros((3,3))
    frames = load_jsons(args.json_folder)
    plot_points(frames, homography)

if __name__ == "__main__":
    parse_args()
    main()