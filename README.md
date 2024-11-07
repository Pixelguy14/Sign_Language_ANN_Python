# Sign_Language_ANN_Python
Leveraging OpenCV, MediaPipe, and Neural Networks for Sign Language Interpretation for Python

Part 1. Extraction of the characteristics form a dataset.

dataset extracted from the next reference:
mrgeislinger (2015, August 12). ASL Fingerspelling Images (RGB & Depth). [Dataset]. Kaggle. https://www.kaggle.com/datasets/mrgeislinger/asl-rgb-depth-fingerspelling-spelling-it-out

hand_landmarks.csv is a file made of 64 characteristics and one tag related to the alphabet letter. each row represents a recognized hand from the dataset and each collum is a coordinate of a hand landmark.
hand_landmarks.csv was created by running multiple instances of sign_language_images_to_csv.py, where each instance corresponed to a diferent folder containing all alphabet letters.