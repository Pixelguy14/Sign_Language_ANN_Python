'''
Predict sign language in real time using MediaPipe, OpenCV
and the trained model of the backpropagation neural network

pip install mediapipe
pip install opencv-python
pip install onnxruntime
'''

import cv2 # OpenCV
import mediapipe as mp # Hand detection
import numpy as np
import onnxruntime as rt # Load onnx model

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load model
sess = rt.InferenceSession('trained_model.onnx')

# Function to preprocess landmarks for the model, 
# turning them to array of float with one row and n ammount of collumns
def preprocess_landmarks(hand_landmarks):
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.extend([landmark.x, landmark.y, landmark.z])
    processed_landmarks = np.array(landmarks).reshape(1, -1).astype(np.float32)

    #print("Processed Landmarks:", processed_landmarks)
    return processed_landmarks

# Function to predict hand gesture using approach similar to trained_model_test.py
def predict_gesture(landmarks):
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    pred = sess.run([output_name], {input_name: landmarks})[0]
    #print("Predicted:", pred)
    #predicted_label = np.argmax(pred) # Ensure predictions are mapped correctly
    predicted_label = labels.get(pred[0])
    #print("Predicted Label:", predicted_label)
    return predicted_label

# Number of class to letter label mapping
labels = {
    0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g",
    7: "h", 8: "i", 9: "k", 10: "l", 11: "m", 12: "n", 13: "o",
    14: "p", 15: "q", 16: "r", 17: "s", 18: "t", 19: "u", 20: "v",
    21: "w", 22: "x", 23: "y"
}

# Initialize webcam
cap = cv2.VideoCapture(0)

# Process hands and detect landmarks
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1) # 1  = horizontal flip ; 0 =  vertical flip
        # re_frame = cv2.resize(frame, (82,117)) # size of the training images
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = hands.process(rgb_frame)

        # Draw hand landmarks on the frame and predict gesture
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Preprocess landmarks and predict gesture
                # we turn the landmarks to a number array 
                # then we process and change the prediction to corresponding label
                landmarks = preprocess_landmarks(hand_landmarks)
                gesture = predict_gesture(landmarks)
                gesture_text = f"Gesture: {gesture}"

                # Display the predicted gesture on the frame
                # frame the text will be on
                # text that's going to be displayed
                # 10, 30 are the coords for the text
                # font
                # font scale (size)
                # color in BGR
                # thickness of the font
                # antialliased text (smooth edges)
                cv2.putText(frame, gesture_text, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Hand Gesture Detection', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
