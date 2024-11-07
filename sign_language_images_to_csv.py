'''
Read Hand Gestures Using MediaPipe and OpenCV to load in csv

pip install mediapipe
pip install opencv-python
'''

# Import libraries
import cv2
import mediapipe as mp
import numpy as np
import csv
import os

print(f"Current directory: {os.getcwd()}")

# Initialize tools
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Save Landmarks to CSV
def save_landmarks_to_csv(landmarks, folder_name, filename='hand_landmarks.csv'):
    ##print("hand recognized correctly")
    try:
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            row = []
            for landmark in landmarks.landmark:
                row.extend([landmark.x, landmark.y, landmark.z])
            row.append(folder_name)
            writer.writerow(row)
        print(f"Successfully saved landmarks to {filename}")
    except Exception as e:
        print(f"Error saving landmarks to {filename}: {e}")

# Read images from folders and process
# base_path is the folder containing folders of the rest of the letters
# we can change base_path to read folders with different names but must be in root
def process_images_from_folders(base_path):
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    ) as hands:
        for folder_name in os.listdir(base_path):
            folder_path = os.path.join(base_path, folder_name)
            if os.path.isdir(folder_path):
                for image_name in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image_name)
                    if image_path.endswith(('jpg', 'jpeg', 'png')):
                        ##print(f"Processing {image_path}")
                        # Read the image
                        image = cv2.imread(image_path)
                        if image is None:
                            ##print(f"Failed to read {image_path}")
                            continue
                        
                        # Convert the BGR image to RGB
                        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        # Process the image
                        # process is part of mediapipe and we use it to detect hands
                        results = hands.process(rgb_image)
                        
                        # Draw hand landmarks on the image and save to CSV only
                        # when it dettects hands in the images
                        if results.multi_hand_landmarks:
                            for hand_landmarks in results.multi_hand_landmarks:
                                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                                # Save landmarks to CSV
                                save_landmarks_to_csv(hand_landmarks, folder_name) 
                        # if we want to see the images it's reading we display
                        # the next block of code, but it will flash open a lot
                        # of images so we better keep it closed. 
                        '''
                        # Display the image
                        cv2.imshow('Hand Gesture Detection', image)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

    cv2.destroyAllWindows()
'''

# Run the processing
# base_path is the folder where all the imagesets of the letters will be contained.
# dataset5 contains 5 folders, A, B, C, D E, change the letter according to the csv
# you'll be creating
process_images_from_folders(base_path = 'A')
