import cv2
import mediapipe as mp

# Load the MediaPipe Face Detection model
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection()

# Open the video file
video_capture = cv2.VideoCapture('video/5.mp4')

# Get the video dimensions
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_capture.get(cv2.CAP_PROP_FPS))

# Create a video writer to save the processed video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

# Loop through each frame of the video
while True:
    ret, frame = video_capture.read()

    if not ret:
        break

    # Convert the frame to RGB format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run face detection on the frame
    results = face_detection.process(frame)

    if results.detections:
    # Draw the detected faces on the frame
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, c = frame.shape
            xmin = int(bbox.xmin * w)
            ymin = int(bbox.ymin * h)
            xmax = int((bbox.xmin + bbox.width) * w)
            ymax = int((bbox.ymin + bbox.height) * h)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            

        # Write the processed frame to the output video
        out.write(frame)

        # Display the processed frame
        cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
face_detection.close()
video_capture.release()
out.release()
cv2.destroyAllWindows()
