# this script logs data from the web cam on an nvidia jetson tx2
import cv2

print("The opencv version is {}".format(cv2.__version__))

# this is the formatting string required to specify the webcam
cap = cv2.VideoCapture('nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480,format=(string)I420, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink')



ret, frame = cap.read();
if not ret:
    raise ValueError("caputure failed to open")

frame_height, frame_width, _ = frame.shape
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

# set the index for writing out frames
i = 0
while(ret):
	# read from the webcam
	ret, frame = cap.read();
	# show the image
        cv2.imshow("video stream", frame) 
	# write the image
	cv2.imwrite('../data/new_video/{:04d}.png'.format(i), frame)
	# write the image to a video file
	out.write(frame)
        print('reading the {}th frame'.format(i))
	
	# quit if the person hits q on the imshow windeo
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	# increment the index
        i += 1
