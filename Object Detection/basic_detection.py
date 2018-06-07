import numpy as np
import argparse
import cv2
import datetime
from imutils.video import FPS
import timeit

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=False, help="path to input image", default='MobileNet_SSD/images/example_01.jpg')
ap.add_argument("-p", "--prototxt", required=False, default='MobileNet_SSD/MobileNetSSD_deploy.prototxt.txt', help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=False, default='MobileNet_SSD/MobileNetSSD_deploy.caffemodel', help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
# args = {
#     'model': 'MobileNet_SSD/MobileNetSSD_deploy.caffemodel',
#     'prototxt': 'MobileNet_SSD/MobileNetSSD_deploy.prototxt.txt',
#     'image': 'MobileNet_SSD/images/example_01.jpg',
#     'confidence': 0.2
# }
import time

if args.get("video", None) is None:
    camera = cv2.VideoCapture(0)
    time.sleep(0.25)
    # otherwise, we are reading from a video file
else:
    camera = cv2.VideoCapture(args["video"])

fps = FPS().start()

total_detection_time, iterations = 0, 0

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


while True and iterations < 500:
    # grab the current frame and initialize the occupied/unoccupied
    # text
    (grabbed, frame) = camera.read()
    (h, w) = frame.shape[:2]

    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if not grabbed:
        break
 
    # resize the frame, convert it to grayscale, and blur it
    # frame = imutils.resize(frame, width=500)

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    start = time.time()
    detections = net.forward()
    end = time.time()
    time_taken = end - start
    total_detection_time += time_taken
    iterations += 1
    # print('[INFO] detection time: ' + str(time_taken))

    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        # print(confidence)
        # print(type(confidence))
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # display the prediction
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            # print("[INFO] {}".format(label))
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # draw the text and timestamp on the frame
    # cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
    #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.imshow('Object Detection', frame)
 
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break
    fps.update()
# cleanup the camera and close any open windows
fps.stop()
print('[INFO] elapsed time: ' + str(fps.elapsed()))
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
print("[INFO] approx. time taken per detection: {:.9f}, iterations: {}".format(total_detection_time/iterations, iterations))

camera.release()
cv2.destroyAllWindows()

# cv2.imshow("Output", image)
# MobileNet_SSD/images

# print(cv2.imwrite(args['image'].split('/')[0] + '/outputs/' + args['image'].split('/')[-1], image))
