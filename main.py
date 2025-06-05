import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,  # Model selection (0 for close-range, 1 for far-range)
    min_detection_confidence=0.5  # Minimum confidence value for detection
)

# Initialize camera
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

while True:
    # Read frame from camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect faces
    results = face_detection.process(rgb_frame)

    # Draw face detections
    if results.detections:
        for detection in results.detections:
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

    # Display the frame
    cv2.imshow('Face Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
cv2.destroyAllWindows()
face_detection.close()