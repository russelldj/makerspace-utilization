import json
import collections

ORDERED_KEYPOINTS = ["Nose",
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
"RHeel"]

Keypoint = collections.namedtuple('Keypoint', 'x y conf')


def parse_json(json_data):
    people_keypoints = [d["pose_keypoints_2d"] for d in json_data["people"]]
    # this is now a list of 75-length lists
    simplified_people = list()
    for person in people_keypoints:
        output = dict() #TODO name better
        for i, keypoint in enumerate(ORDERED_KEYPOINTS):
            point = Keypoint(x=person[3*i], y=person[3*i+1], conf=person[3*i+2])
            output[keypoint] = point
        simplified_people.append(output)
    return simplified_people
             





