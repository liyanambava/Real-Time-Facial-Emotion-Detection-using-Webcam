import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model_path = r"C:\Users\LIYANA\OneDrive\Documents\Projects\best_model.h5"
emotion_model = load_model(model_path)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#to detect emotion in frames
def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray_resized = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        # Convert grayscale to RGB by stacking the image along the channel axis
        roi_rgb = np.stack((roi_gray_resized,) * 3, axis=-1)
        
        roi = roi_rgb.astype('float32') / 255.0  # Normalize
        roi = np.expand_dims(roi, axis=0)        # Add batch dimension (1, 48, 48, 3)
        
        prediction = emotion_model.predict(roi)[0]
        max_index = int(np.argmax(prediction))
        emotion = emotion_labels[max_index]
        
        label_position = (x, y-10)
        cv2.putText(frame, emotion, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame

#webcam start frame - press 'q' to quit webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = detect_emotion(frame)
    cv2.imshow('Emotion Detector', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
