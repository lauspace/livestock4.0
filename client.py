# import libraries
import argparse 
import time
import glob 

import cv2 
import numpy as np 
import paho.mqtt.client as mqtt 

# parser argument definition
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input image file")
parser.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output image file. Write only the name, without extension.")
parser.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
parser.add_argument("-t", "--threshold", type=float, default=0.4,
	help="threshold for non maxima supression")

args = parser.parse_args()

# variable initialitation
CONFIDENCE_THRESHOLD = args.confidence
NMS_THRESHOLD = args.threshold
img_path = args.input

# load weights, labels and config file: DL model definition
weights = glob.glob("dl_model/*.weights")[0]
labels = glob.glob("dl_model/*.txt")[0]
cfg = glob.glob("dl_model/*.cfg")[0]

print("You are now using {} weights ,{} configs and {} labels.".format(weights, cfg, labels))

# variable assignation
lbls = list()
with open(labels, "r") as f:
    lbls = [c.strip() for c in f.readlines()]

COLORS = np.random.randint(0, 255, size=(len(lbls), 3), dtype="uint8")

# save model configuration in net variable using CUDA
net = cv2.dnn.readNetFromDarknet(cfg, weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# layer definition
layer = net.getLayerNames()
layer = [layer[i - 1] for i in net.getUnconnectedOutLayers()]

# function to capture image with pi camera (not available yet)
def capture_image():
    # create camera object
    camera = PiCamera()
    # camera needs 2 sec to inizialize it
    time.sleep(2)
    
    # capture image
    path = ""
    camera.capture(path)
    
    return path

# detection function
def detect(imgpath, nn):
    # read image and save dimensions
    image = cv2.imread(imgpath)
    (H, W) = image.shape[:2]
    
    # define image parameters
    blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), swapRB=True, crop=False)

    # inference
    nn.setInput(blob)
    start_time = time.time()
    layer_outs = nn.forward(layer)
    end_time = time.time()

    # create bounding boxes, confidence and class_ids variables
    boxes = list()
    confidences = list()
    class_ids = list()
    
    for output in layer_outs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
	    
	    # save image if confidence and id is useful for us
            if confidence > CONFIDENCE_THRESHOLD and class_id == 0:
                box = detection[0:4] * np.array([W, H, W, H])
                (center_x, center_y, width, height) = box.astype("int")

                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return len(idxs)
        
# send message function with mqtt protocol 
def send_message(mssg):
    client = mqtt.Client()
    client.connect("192.168.1.108", 1883, 60)
    client.publish("sensor1", mssg)

detections = detect(img_path, net)
send_message(detections)  
