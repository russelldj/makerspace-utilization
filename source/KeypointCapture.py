# KeypointCapture.py
# For storing captured keypoints from OpenPose

import numpy as np
import glob
import json
import copy
import pdb

# Globals that define the order of the keypoints
ORDERED_KEYPOINTS_BODY = [
"Nose",
"Neck",
"RShoulder",
"RElbow",
"RWrist",
"LShoulder",
"LElbow",
"LWrist",
"MidHip",
"RHip",
"RKnee",
"RAnkle",
"LHip",
"LKnee",
"LAnkle",
"REye",
"LEye",
"REar",
"LEar",
"LBigToe",
"LSmallToe",
"LHeel",
"RBigToe",
"RSmallToe",
"RHeel"
]

ORDERED_KEYPOINTS_HAND =[
    "Wrist",
    "Thumb1",
    "Thumb2",
    "Thumb3",
    "Thumb4",
    "Index1",
    "Index2",
    "Index3",
    "Index4",
    "Middle1",
    "Middle2",
    "Middle3",
    "Middle4",
    "Ring1",
    "Ring2",
    "Ring3",
    "Ring4",
    "Pinky1",
    "Pinky2",
    "Pinky3",
    "Pinky4"
]



#POINT_TUPLE = collections.namedtuple('Keypoint', 'x y conf')

# Object to store the keypoints for a single capture
# All keypoint variables are lists of dictionaries with the keys as ORDERED_KEYPOINTS_*
#   and the values as [x, y, conf]
class KeypointCapture:
    def __init__(self):
        self.capture_name = ""
        self.capture_id = ""
        self.right_hand_keypoints = {}
        self.left_hand_keypoints = {}
        self.body_keypoints = {}
        self.num_frames = 0

    def GetKeypointOrdering(self):
        return copy.copy(ORDERED_KEYPOINTS_BODY) + ["Left_" + x for x in ORDERED_KEYPOINTS_HAND] + ["Right_" + x for x in ORDERED_KEYPOINTS_HAND]

    def GetAllKeypointsList(self):
        """
        returns all of the keypoints as a concatenated list
        """
        all_keypoints = [copy.copy(self.body_keypoints), copy.copy(self.left_hand_keypoints), copy.copy(self.right_hand_keypoints)]

    def GetSingleFrameAsOneDict(self, frame):
        """
        Return the keypoints as a dict indexed by the joint name (prefaced by Right or Left for hands) with the value as the current list
        This should be used for copying only

        args
        ----------
        frame : int
            The index in the capture you wish to get a copy of, not error checked
        """

        output_dict = {}
        for k,v in self.body_keypoints.items():
            output_dict[k] = v[frame]

        for k,v in self.left_hand_keypoints.items():
            output_dict[k] = v[frame]

        for k,v in self.right_hand_keypoints.items():
            output_dict[k] = v[frame]

        return output_dict

    def GetKeypointsAsOneDict(self):
        """
        Returns all the keypoints combined into a single dict

        args
        ----------
        """

        output_dict = {}
        for k,v in self.body_keypoints.items():
            output_dict[k] = v

        for k,v in self.left_hand_keypoints.items():
            output_dict[k] = v

        for k,v in self.right_hand_keypoints.items():
            output_dict[k] = v

        return output_dict

    # Gets a deep of the current KeypointCapture instance
    def DeepCopy(self):
        k_copy = KeypointCapture()
        k_copy.capture_name = copy.copy(self.capture_name)
        k_copy.capture_id = copy.copy(self.capture_id)
        k_copy.num_frames = self.num_frames

        for k,v in self.body_keypoints.items():
            k_copy.body_keypoints[k] = [None] * self.num_frames
            for frame in range(self.num_frames):
                k_copy.body_keypoints[k][frame] = copy.copy(v[frame])

        for k,v in self.right_hand_keypoints.items():
            k_copy.right_hand_keypoints[k] = [None] * self.num_frames
            for frame in range(self.num_frames):
                k_copy.right_hand_keypoints[k][frame] = copy.copy(v[frame])

        for k,v in self.left_hand_keypoints.items():
            k_copy.left_hand_keypoints[k] = [None] * self.num_frames
            for frame in range(self.num_frames):
                k_copy.left_hand_keypoints[k][frame] = copy.copy(v[frame])

        return k_copy

    #def GetKeypointsList(self, frame):
    #    keypoint_dict = self.GetFrameKeypointsDict(frame)
    #    for key in sorted(keypoint_dict.keys())


# Parses an entire folder of 2D json keypoint frames
def Read2DJsonPath(jsonFolder, captureName, captureId):
    if jsonFolder[-1] != "/":
        jsonFolder += "/"
    files = sorted(glob.glob("{}*".format(jsonFolder)))
    frames = list()
    for file_ in files:
        with open(file_, 'r') as json_file:
            json_data = json.loads(json_file.read())
            frames.append(json_data)

    keypoint_capture = Parse2DJsonFrames(frames, captureName, captureId)
    return keypoint_capture


# Parses a 2D Json collection of frames into a KeypointCapture
def Parse2DJsonFrames(jsonFrames, captureName, captureId, hasHands=False):
    # Initializing keypoint object and dicts for performance purposes
    keypoint_capture = KeypointCapture()

    body_dict = {}
    right_hand_dict = {}
    left_hand_dict = {}
    for keypoint in ORDERED_KEYPOINTS_BODY:
        body_dict[keypoint] = [None] * len(jsonFrames)
    for keypoint in ORDERED_KEYPOINTS_HAND:
        right_hand_dict["Right_" + keypoint] = [None] * len(jsonFrames)
        left_hand_dict["Left_" + keypoint] = [None] * len(jsonFrames)

    for i in range(len(jsonFrames)):
        json_frame = jsonFrames[i]

        # Grabbing our target 2D keypoints
        # TODO: Generalize to any number of people in scene
        # body_keypoints = [d["pose_keypoints_2d"] for d in json_frame["people"]]
        # left_hand_keypoints = [d["hand_left_keypoints_2d"] for d in json_frame["people"]]
        # right_hand_keypoints = [d["hand_right_keypoints_2d"] for d in json_frame["people"]]
        if len(json_frame["people"]) > 0:
            person = json_frame["people"][0]
            body_keypoints = person["pose_keypoints_2d"]
            left_hand_keypoints = person["hand_left_keypoints_2d"]
            right_hand_keypoints = person["hand_right_keypoints_2d"]

            # Now adding to keypoint object

            # Adding body points
            for j in range(len(ORDERED_KEYPOINTS_BODY)):
                print("j is {}".format(j))
                x_body = body_keypoints[3*j]
                y_body = body_keypoints[3*j+1]
                conf_body = body_keypoints[3*j+2]
                body_dict[ORDERED_KEYPOINTS_BODY[j]][i] = [x_body, y_body, conf_body]

                # Adding hand points
                if hasHands:
                    for j in range(len(ORDERED_KEYPOINTS_HAND)):
                        x_left = left_hand_keypoints[3*j]
                        y_left = left_hand_keypoints[3*j+1]
                        conf_left = left_hand_keypoints[3*j+2]
                        left_hand_dict["Left_" + ORDERED_KEYPOINTS_HAND[j]][i] = [x_left, y_left, conf_left]

                        x_right = right_hand_keypoints[3*j]
                        y_right = right_hand_keypoints[3*j+1]
                        conf_right = right_hand_keypoints[3*j+2]
                        right_hand_dict["Right_" + ORDERED_KEYPOINTS_HAND[j]][i] = [x_right, y_right, conf_right]
        else:
            pass
            #TODO dermine if we need to do anything

                

    keypoint_capture.capture_name = captureName
    keypoint_capture.capture_id = captureId
    keypoint_capture.body_keypoints = body_dict
    keypoint_capture.right_hand_keypoints = right_hand_dict
    keypoint_capture.left_hand_keypoints = left_hand_dict
    keypoint_capture.num_frames = len(jsonFrames)
    return keypoint_capture
