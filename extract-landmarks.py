import cv2
import mediapipe as mp
import pickle
import numpy as np
import os

# Initialize MediaPipe Face Mesh (for landmarks)
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,  # For simplicity, we'll detect only one face
    refine_landmarks=True,  # Better landmarks around eyes and lips
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    static_image_mode=True
)

#Root directory
root_dir = './images'

#Directory to the dataset
subsets = ["train", "validation"]

#To store the landmarks
landmark_data = []

for subset in subsets:
  subset_path = os.path.join(root_dir, subset)

  for label in os.listdir(subset_path):
    emotion_path = os.path.join(subset_path, label)
    if not os.path.isdir(emotion_path):
      continue

    for img_name in os.listdir(emotion_path):
      img_path = os.path.join(emotion_path, img_name)

      image = cv2.imread(img_path)
      if image is None:
        continue

      #Set image to RGB
      rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


      rgb_image.flags.writeable = False

      # Process the image for facial landmarks
      mesh_results = face_mesh.process(rgb_image)

      # Set image as writeable again
      rgb_image.flags.writeable = True

      if mesh_results.multi_face_landmarks:
          for face_landmarks in mesh_results.multi_face_landmarks:
            landmarks = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
            landmark_data.append({
                'subset': subset,
                'emotion': label,
                'image_path': img_path,
                'landmarks': landmarks
            })
            
      print(f"landmarks for {img_path} has been extracted")

print(f"Extracted landmarks from {len(landmark_data)} images.")

# Save to file (optional)
with open("facial_landmarks_data.pkl", "wb") as f:
    pickle.dump(landmark_data, f)

