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


# Initialize tools
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Save Landmarks to CSV
def save_landmarks_to_csv(landmarks, folder_name, filename='hand_landmarks.csv'):
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

# create csv to test one image
# mode 'w' makes it so it overwrites the image creation
def overwrite_landmarks_csv(landmarks, prediction='unknown', filename='predict_landmarks.csv'):
    try:
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            row = []
            for landmark in landmarks.landmark:
                row.extend([landmark.x, landmark.y, landmark.z])
            row.append(prediction) #last collumn will allways be prediction
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
        max_num_hands=1,
        min_detection_confidence=0.1
    ) as hands:
        for folder_name in os.listdir(base_path):
            folder_path = os.path.join(base_path, folder_name)
            if os.path.isdir(folder_path):
                for image_name in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image_name)
                    if image_path.endswith(('jpg', 'jpeg', 'png')):
                        # Read the image
                        image = cv2.imread(image_path)
                        if image is None:
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

# A function that reads a specific folder and gets only one image, 
# it processes the hands detected in it and sends it to write a csv file 
# with that data.
def process_image_from_route(path, image_name_to_predict):
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:
        for image_name in os.listdir(path):
            image_path = os.path.join(path, image_name)
            if image_path.endswith(('jpg', 'jpeg', 'png')) and image_name == image_name_to_predict:
                print(f"Processing {image_name_to_predict}")
                # Read the image
                image = cv2.imread(image_path)
                if image is None:
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
                        overwrite_landmarks_csv(hand_landmarks) 
                    # Save the image with landmarks in the same folder 
                    output_image_path = os.path.join(path, f"processed_{image_name}") 
                    cv2.imwrite(output_image_path, image) 
                    print(f"Saved processed image to {output_image_path}")
                # Display the image
                cv2.imshow('Hand Gesture Detection', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cv2.destroyAllWindows()

# Run the processing
# base_path is the folder where all the imagesets of the letters will be contained.
# dataset5 contains 5 folders, A, B, C, D E, change the letter according to the csv
# you'll be creating

# Run the processing to detect only the image in the prediction folder that has
# a specific name
