import cv2
import os
from utils import detect_faces, analyze_face

def process_video(input_video, output_video="output/annotated_video.mp4"):
    """Process an uploaded video file, annotate it, and save the output."""
    cap = cv2.VideoCapture("video3.mp4")
    #cap = cv2.VideoCapture(0)
    #^^ for webcam detection


    
    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define video writer to save output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame, max_faces=10)

        for face in faces:
            x, y, w, h = face
            gender, ethnicity = analyze_face(frame, face)

            if gender and ethnicity:
                label = f"{gender}, {ethnicity}"
                color = (0, 255, 0)  # Green bounding box

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Put label
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(frame)  # Save the annotated frame

        cv2.imshow("Processing Video...", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to stop
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Annotated video saved as: {output_video}")

def process_webcam():
    """Process real-time video from webcam."""
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame, max_faces=10)

        for face in faces:
            x, y, w, h = face
            gender, ethnicity = analyze_face(frame, face)

            if gender and ethnicity:
                label = f"{gender}, {ethnicity}"
                color = (0, 255, 0)  # Green bounding box

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Put label
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Webcam Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Ask user for mode
    mode = input("Enter 'w' for webcam or 'v' to upload a video: ").strip().lower()

    if mode == 'w':
        process_webcam()
    elif mode == 'v':
        video_path = input("Enter video file path (e.g., input_videos/video.mp4): ").strip()
        if os.path.exists(video_path):
            process_video(video_path)
        else:
            print("Error: File not found!")
    else:
        print("Invalid input. Please enter 'w' or 'v'.")
