from flask import Flask, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load Haar Cascade for face and eye detection
face_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_eye.xml')

# Load the emotion detection model
emotion_model = load_model('models/_mini_XCEPTION.102-0.66.hdf5')

app = Flask(__name__)
@app.route('/camera', methods=['POST'])
def process_image():
    try:
        snapshot_file = request.files['snapshot']
        if snapshot_file:
            # Save the snapshot to a specific path
            snapshot_path = 'pics/snapshot.jpg'
            snapshot_file.save(snapshot_path)

            # Load the image using OpenCV
            image = cv2.imread(snapshot_path)

            # Perform face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

            # Iterate over detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = image[y:y + h, x:x + w]

                # Perform eye detection within each detected face region
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            # Perform emotion detection
            # You'll need to preprocess and resize the image to match the input size expected by the emotion model
            # Then, use the model to predict emotions on the face regions

            # Save the annotated image with detection results
            cv2.imwrite('pics/annotated_snapshot.jpg', image)

            return 'Image processed successfully', 200
        else:
            return 'Snapshot file not found', 400
    except Exception as e:
        return f'Error processing image: {str(e)}', 500

if __name__ == "__main__":
    app.run(debug=True)