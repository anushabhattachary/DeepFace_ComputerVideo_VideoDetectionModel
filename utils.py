import cv2
from deepface import DeepFace

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_faces(frame, max_faces=10):
    """Detect faces in a frame using OpenCV."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    return faces[:max_faces]  # Limit the number of detected faces

def analyze_face(frame, face):
    """Analyze a face region using DeepFace for gender and ethnicity."""
    x, y, w, h = face
    face_roi = frame[y:y+h, x:x+w]  # Crop the face region

    try:
        analysis = DeepFace.analyze(face_roi, actions=['gender', 'race'], enforce_detection=False)
        gender = analysis[0]['dominant_gender']
        ethnicity = analysis[0]['dominant_race']
        return gender, ethnicity
    except Exception as e:
        print(f"DeepFace error: {e}")
        return None, None
