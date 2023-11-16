import os
import cv2
import numpy as np
from PIL import Image
from cvzone.HandTrackingModule import HandDetector
from tensorflow import keras

loadmodel = keras.models.load_model('keras/keras_model.h5')

cap = cv2.VideoCapture(0)
detect = HandDetector(maxHands=1, detectionCon=0.8)
count = 0
extra = 30

# Load the labels from the text file
with open("keras/labels.txt", "r") as file:
    labels = file.read().splitlines()

while True:
    _, frame = cap.read()
    hands, frame = detect.findHands(frame)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        roi = frame[y - extra:y + h + extra, x - extra:x + w + extra]
        roi = cv2.resize(roi, (224, 224))  # Resize roi to match model input shape
        roi = np.expand_dims(roi, axis=0)  # Add batch dimension
        prediction = loadmodel.predict(roi)

        # Get the predicted label index
        predicted_index = np.argmax(prediction)

        # Get the predicted label from the loaded labels
        predicted_label = labels[predicted_index]

        # Write the predicted label on the output frame
        cv2.putText(frame, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Image", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
