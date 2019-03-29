import matplotlib
matplotlib.use("Agg") # this turns off visualizations
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import animation
import numpy as np
#import types
from PIL import Image
import cv2
import KeypointCapture
import pdb

REF_IMAGE_FNAME = "black.png"
VIDEO_OUTPUT_FNAME = "keypoint_video.avi"
REF_VIDEO_FNAME = "video1.avi"

class KeypointVisualization:
    def __init__(self):
        self.FFMpegWriter = manimation.writers['ffmpeg']
        self.metadata = dict(title='Movie Test', artist='Matplotlib',
                            comment='Movie support!')
        self.writer = self.FFMpegWriter(fps=15, metadata=self.metadata)
        self.fig = plt.figure(figsize=(16,10))
        self.canvas = FigureCanvas(self.fig)


    def TestPlot(self):
        pass
        keypoint_capture = KeypointCapture.Read2DJsonPath("video1_json", "0", "0")
        self.WritePointsToVideo(keypoint_capture)


    def WritePointsToVideo(self, keypoint_capture, video_output_file=VIDEO_OUTPUT_FNAME, reference_video=REF_VIDEO_FNAME, FRAME_RATE=30, num_frames=np.inf):
        """
        Writes the overlayed keypoints to a video

        args
        ----------
        keypoint_capture : KeypointCapture
            The keypoints
        video_output_file : str
            Where to write the visualization
        reference_video : str
            The filename of the video the keypoints were computed from
        FRAME_RATE : int
            The output framerate of the written video
        num_frames : int
            Will write out only the first num_frames frames

        return
        ----------
        None
        """
        pdb.set_trace()
        ordered_keys = keypoint_capture.GetKeypointOrdering()
        video_cap = cv2.VideoCapture(reference_video)

        if not video_cap.isOpened():
            print("failed to open video")
        # set up the video writer
        num_frames = min(num_frames, keypoint_capture.num_frames)

        keypoint_dict = keypoint_capture.GetKeypointsAsOneDict()
        pdb.set_trace()
        for i in range(num_frames):
            print("Processing frame {} of {}".format(i, num_frames))
            ret, im = video_cap.read()
            vis_img = self.PlotSingleFrameOpenCV(keypoint_dict, i, ordered_keys, im)
            if i == 0:
                # create the video writer

                image_shape = vis_img.shape[0:2]
                vid_writer = cv2.VideoWriter(video_output_file, cv2.VideoWriter_fourcc('M','J','P','G'), FRAME_RATE, (image_shape[1], image_shape[0]))

            vid_writer.write(vis_img)
        vid_writer.release()
        #animation = self.camera.animate()
        #animation.save(video_output_file)

    # TODO: Fix for new format
    # def PlotSingleFrame(self, keypoint_dict, ordered_keys, im):
    #     plt.clf()
    #     plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    #     num_points = len(ordered_keys)
    #     color=plt.cm.rainbow(np.linspace(0,1,len(ordered_keys)))
    #
    #     def getKeypointSize(i):
    #         # Makes the hands be plotted smaller
    #         return 20 if i < 25 else 2
    #
    #     for i, key in enumerate(ordered_keys):
    #         x, y, conf = keypoint_dict[key]
    #         plt.scatter(x, y, c=[color[i]], s=getKeypointSize(i))
    #     self.canvas.draw()
    #     s, (width, height) = self.canvas.print_to_buffer()
    #     image = np.fromstring( self.canvas.tostring_rgb(), dtype='uint8').reshape((height, width, 3))
    #     return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def PlotSingleFrameOpenCV(self, keypoint_dict, frame, ordered_keys, im):
        num_points = len(ordered_keys)
        colors=plt.cm.rainbow(np.linspace(0,1,len(ordered_keys)))

        def getKeypointSize(i):
            # Makes the hands be plotted smaller
            return 7 if i < 25 else 3

        for i, key in enumerate(ordered_keys):
            xyc = keypoint_dict[key][frame]
            if(type(xyc) is list):
                x, y, conf = xyc
                color = [int(x * 255) for x in colors[i,0:3]]
                cv2.circle(im, (int(x), int(y)), 1 , color, getKeypointSize(i))
        return im


if __name__ == "__main__":
    #KeypointVisualization().TestWriteToVideo()
    KeypointVisualization().TestPlot()
