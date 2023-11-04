import cv2
import face_recognition
import tensorflow as tf
import numpy as np

# Load the pre-trained gender classification model
gender_model = tf.keras.models.load_model('gender_model.h5')

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Find faces in the frame
    face_locations = face_recognition.face_locations(frame)

    for top, right, bottom, left in face_locations:
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Extract the face region
        face = frame[top:bottom, left:right]

        # Resize the face for gender classification
        face = cv2.resize(face, (64, 64))
        face = np.expand_dims(face, axis=0)

        # Predict the gender
        gender_prob = gender_model.predict(face)[0]
        gender = "Male" if gender_prob[0] > 0.5 else "Female"

        # Display the gender
        cv2.putText(frame, f'Gender: {gender}', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the frame with boxes and gender labels
    cv2.imshow('Face Detection and Gender Classification', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press Esc to exit
        break

cap.release()
cv2.destroyAllWindows()
