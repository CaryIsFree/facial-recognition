import cv2
import mediapipe as mp
import time

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
print_interval = 3  # Print landmark data every 3 seconds

while True:
    # Read frame from camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # To improve performance, mark the image as not writeable to pass by reference
    rgb_frame.flags.writeable = False
    
    # Process the frame for face detection
    detection_results = face_detection.process(rgb_frame)
    
    # Process the frame for facial landmarks
    mesh_results = face_mesh.process(rgb_frame)
    
    # Set image as writeable again
    rgb_frame.flags.writeable = True
    
    # Convert back to BGR for display
    frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

    # Draw face detections (from week 2)
    if detection_results.detections:
        for detection in detection_results.detections:
            # Draw the face detection box
            mp_drawing.draw_detection(frame, detection)

            # Get detection confidence and display it
            confidence = detection.score[0]
            bbox = detection.location_data.relative_bounding_box
            h, w, c = frame.shape
            x, y = int(bbox.xmin * w), int(bbox.ymin * h)
            
            # Display confidence score
            confidence_text = f'Confidence: {confidence:.2f}'
            cv2.putText(frame, confidence_text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # TODO: Draw facial landmarks
    # If facial landmarks are detected, visualize them on the frame
    if mesh_results.multi_face_landmarks:
        for face_landmarks in mesh_results.multi_face_landmarks:
            # Draw the landmarks of the face
            mp_drawing.draw_landmarks(
                image = frame,
                landmark_list = face_landmarks,
                connections = mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec = None,
                connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

            # Draw the contours of the face
            mp_drawing.draw_landmarks(
                image = frame,
                landmark_list = face_landmarks,
                connections = mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec = None,
                connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            
            # Print landmark data every 3 seconds
            current_time = time.time()
            if current_time - last_print_time >= print_interval:
                print("\n--- Facial Landmarks Data ---")
                print(f"Total landmarks: {len(face_landmarks.landmark)}")
                
                # Print the first 5 landmarks as an example
                print("Sample landmark positions (normalized coordinates):")
                for i, landmark in enumerate(face_landmarks.landmark[:5]):
                    print(f"Landmark {i}: x={landmark.x:.4f}, y={landmark.y:.4f}, z={landmark.z:.4f}")
                
                last_print_time = current_time

    # Display the frame
    cv2.imshow('Face Detection & Landmarks', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
cv2.destroyAllWindows()
face_detection.close()
face_mesh.close()