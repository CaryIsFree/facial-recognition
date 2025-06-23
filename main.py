import cv2
import mediapipe as mp
import time
import torch
import torch.nn as nn
import numpy as np

class Classifier(nn.Module):
    def __init__(self, input_size=1434, hidden_size1=256, hidden_size2=128, output_size=7, dropout_prob=0.4):
        """
        Args:
            input_size: Number of input features, 468 landmarks * 3 coords = 1404
            hidden_size: Number of neurons in the hidden layer
            output_size: Number of output classes, 7 for 7 different emotions
        """
        super(Classifier, self).__init__()
        
        # Define the layers
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size1)
        self.activation1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size2)
        self.activation2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)
        self.layer3 = nn.Linear(hidden_size2, output_size)
        
    def forward(self, x):
        """
        Forward pass through the network.
        """
        x = self.layer1(x)
        x = self.batch_norm1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.batch_norm2(x)
        x = self.activation2(x)
        x = self.dropout2(x)
        x = self.layer3(x)
        return x

#Use the gpu to train model instead of cpu
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

#Instantiate model and load weights
model_path = "emotion_model.pt"

emotion_model = Classifier(
    input_size=1434, #478 * 3 = 1434 landmarks
    hidden_size1=256,
    output_size=7,
    dropout_prob=0.3
    ).to(device)

try:
    emotion_model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'emotion_model.pt' exists and the Classifier definition matches the saved model.")
    exit()

emotion_model.eval() # Set model to evaluation mode

# 4. Define emotion mapping (reverse of your training emotion_to_idx)
emotion_to_idx_train = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprise": 6
}
idx_to_emotion = {idx: emotion for emotion, idx in emotion_to_idx_train.items()}

# Initialize MediaPipe Face Mesh (for landmarks)
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize face detection and face mesh
face_detection = mp.solutions.face_detection.FaceDetection(
    model_selection=0,  # Model selection (0 for close-range, 1 for far-range)
    min_detection_confidence=0.5  # Minimum confidence value for detection
)

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,  # For simplicity, we'll detect only one face
    refine_landmarks=True,  # Better landmarks around eyes and lips
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize camera
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Variables for timing the landmark printing
last_print_time = time.time()
print_interval = 1  # Print landmark data every 3 seconds
predicted_emotion_text = "Emotion: ..."

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    
    # Process for facial landmarks
    mesh_results = face_mesh.process(rgb_frame)
    
    rgb_frame.flags.writeable = True # Not strictly necessary if not drawing on rgb_frame
    # frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR) # Convert back if you modified rgb_frame for drawing

    current_time = time.time()
    perform_prediction = (current_time - last_print_time >= print_interval)

    if mesh_results.multi_face_landmarks:
        for face_landmarks in mesh_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

            if perform_prediction:
                # --- Prepare landmarks for model ---
                landmarks_for_model = []
                for lm in face_landmarks.landmark:
                    landmarks_for_model.extend([lm.x, lm.y, lm.z])
                
                # Check if the number of features matches the model's input size
                if len(landmarks_for_model) != 1434:
                    # This can happen if refine_landmarks=False or different MediaPipe version
                    # For now, we'll skip prediction if this happens, but ideally, you'd ensure consistency
                    print(f"Warning: Number of extracted landmark features (1434) does not match model input size. Skipping prediction.")
                    predicted_emotion_text = "Error: Landmark mismatch"
                else:
                    landmarks_tensor = torch.tensor([landmarks_for_model], dtype=torch.float32).to(device) # Add batch dimension

                    # --- Make prediction ---
                    with torch.no_grad(): # No need to calculate gradients
                        outputs = emotion_model(landmarks_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        confidence, predicted_idx = torch.max(probabilities, 1)
                        
                        pred_emotion = idx_to_emotion[predicted_idx.item()]
                        pred_confidence = confidence.item()
                        predicted_emotion_text = f"Emotion: {pred_emotion} ({pred_confidence*100:.1f}%)"
                
                last_predict_time = current_time # Reset time only after attempting prediction
    else:
        # If no face is detected, clear the emotion text or set to "No face"
        if perform_prediction: # Clear old text if it's time to update
             predicted_emotion_text = "Emotion: No face"


    # Display the predicted emotion on the frame
    # Position it somewhere sensible, e.g., top-left
    cv2.putText(frame, predicted_emotion_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Real-time Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# face_detection.close() # Only if you re-enable it
face_mesh.close()