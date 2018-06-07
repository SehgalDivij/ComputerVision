
# coding: utf-8

# In[1]:


import numpy as np
import argparse
import cv2
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False, help="path to input image", default='MobileNet_SSD/images/example_01.jpg')
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


# In[11]:


import time


# In[2]:


# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


# In[3]:


print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


# In[4]:


image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)


# In[5]:


# cv2.imshow('image', image)


# In[6]:


print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()


# In[7]:


#print(detections)
#detections.shape
#detections[0, 0, 0, 2]


# In[24]:


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
        print("[INFO] {}".format(label))
        cv2.rectangle(image, (startX, startY), (endX, endY),
            COLORS[idx], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image, label, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
cv2.imshow(label,image)
cv2.waitKey(0)
print(cv2.imwrite(args['image'].split('/')[0] + '/outputs/' + args['image'].split('/')[-1], image))


# In[21]:


# cv2.imshow("Output", image)
# MobileNet_SSD/images
'a-bc-d-ef-gh'.split('-')[-1]

