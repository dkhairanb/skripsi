# Import packages
import os
import cv2
import numpy as np
import time

import RPi.GPIO as GPIO

from stepper_function import execute_command

from threading import Thread
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter

IR_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(IR_PIN, GPIO.IN)

# Define VideoStream class to handle streaming of video from the Picamera2 in a separate processing thread
class VideoStream:
    """Camera object that controls video streaming from the Picamera2"""
    def __init__(self, resolution=(640, 480), framerate=30):
        self.picam2 = Picamera2()
        self.picam2.preview_configuration.main.size = resolution
        self.picam2.preview_configuration.main.format = "RGB888"
        self.picam2.preview_configuration.controls.FrameRate = framerate
        self.picam2.configure("preview")
        self.picam2.start()
        
        # Read first frame from the stream
        self.frame = self.picam2.capture_array()
        
        # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
        # Start the thread that reads frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                self.picam2.stop()
                return

            # Otherwise, grab the next frame from the stream
            self.frame = self.picam2.capture_array()

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True

# Get path to current working directory
CWD_PATH = os.getcwd()

# Define model and label paths
MODEL_NAME = "custom_model_lite"
GRAPH_NAME = "detect.tflite"  # Assuming your model file name is detect.tflite
LABELMAP_NAME = "labelmap.txt"  # Assuming your label map file name is labelmap.txt

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load the Tensorflow Lite model.
interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# Check output layer name to determine if this model was created with TF2 or TF1
outname = output_details[0]['name']

if 'StatefulPartitionedCall' in outname:  # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:  # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(640, 480), framerate=30).start()
time.sleep(1)


# Function to perform object detection
def perform_object_detection():
    global object_detected

    # Grab frame from video stream
    frame1 = videostream.read()

    # Preprocess frame
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values
    input_data = (np.float32(input_data) - 127.5) / 127.5

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

    # Find the index of the object with the highest confidence score
    max_score_index = np.argmax(scores)

    # Extract the information of the object with the highest confidence score
    max_score = scores[max_score_index]

    # If object is detected with sufficient confidence, print and draw bounding box
    if max_score > 0.7:  # Adjust confidence threshold as needed
        max_score_class = int(classes[max_score_index])
        max_score_label = labels[max_score_class]
        ymin, xmin, ymax, xmax = boxes[max_score_index]
        xmin = int(xmin * frame.shape[1])
        xmax = int(xmax * frame.shape[1])
        ymin = int(ymin * frame.shape[0])
        ymax = int(ymax * frame.shape[0])

        # Print detected object information
        print("Detected object:", max_score_label)
        print("Confidence:", max_score)
        print("Bounding box:", (xmin, ymin, xmax, ymax))

        # Draw bounding box on frame
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(frame, '{}: {:.2f}'.format(max_score_label, max_score), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Set object detected flag
        object_detected = True

        if max_score_label == "Plastic":
            execute_command(1)
        elif max_score_label == "Can":
            execute_command(2)
        elif max_score_label == "Paper":
            execute_command(3)



        # Display frame with detected object
        # cv2.imshow('Object detector', frame)

        # Reset object_detected flag for the next detection
        object_detected = False

# Main loop
while True:
    # Ask for user input to trigger object detection
    user_input = input("Enter 'detect' to perform object detection (or 'q' to quit): ")
    if user_input.strip().lower() == 'q':
        break

    # Check if user wants to perform object detection
    if user_input.strip().lower() == '1':
        perform_object_detection()

    test = "plastic"
        
    # if GPIO.input(IR_PIN) == GPIO.HIGH:
    #     perform_object_detection()
    #     time.sleep(2)

# Clean up
GPIO.cleanup
cv2.destroyAllWindows()
videostream.stop()