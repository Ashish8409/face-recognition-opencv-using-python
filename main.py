import cv2
import face_recognition  

# Load known face encodings and names
known_face_encodings = []
known_face_names = []

# Load known faces and their names here
try:
    known_Ajmal_image = face_recognition.load_image_file("images/ajmal.jpg")
    known_Ashish_image = face_recognition.load_image_file("images/19912ashish.jpg")
    known_Alok_image = face_recognition.load_image_file("images/alok.jpg")
    known_Vidya_bhushan_image = face_recognition.load_image_file("images/vidya bhushan.jpg")
    
    known_Aditya_image = face_recognition.load_image_file("images/aditya.jpg")
    known_Amit_image = face_recognition.load_image_file("images/amit.jpg")
    known_Neeraj_image = face_recognition.load_image_file("images/neeraj.jpg")
    known_Pratik_image = face_recognition.load_image_file("images/pratik.jpg")
    known_Ishan_image = face_recognition.load_image_file("images/ishan.jpg")
    
    # Get the face encodings for the known faces
    known_Ajmal_encoding = face_recognition.face_encodings(known_Ajmal_image)
    known_Ashish_encoding = face_recognition.face_encodings(known_Ashish_image)
    known_Alok_encoding = face_recognition.face_encodings(known_Alok_image)
    known_Vidya_bhushan_encoding = face_recognition.face_encodings(known_Vidya_bhushan_image)
    
    known_Aditya_encoding = face_recognition.face_encodings(known_Aditya_image)
    known_Amit_encoding = face_recognition.face_encodings(known_Amit_image)
    known_Neeraj_encoding = face_recognition.face_encodings(known_Neeraj_image)
    known_pratik_encoding = face_recognition.face_encodings(known_Pratik_image)
    known_Ishan_encoding = face_recognition.face_encodings(known_Ishan_image)

    # Check if encodings were found and append to the lists
    if known_Ajmal_encoding:
        known_face_encodings.append(known_Ajmal_encoding[0])
        known_face_names.append("Ajmal")

    if known_Ashish_encoding:
        known_face_encodings.append(known_Ashish_encoding[0])
        known_face_names.append("Ashish")

    if known_Alok_encoding:
        known_face_encodings.append(known_Alok_encoding[0])
        known_face_names.append("Alok")
        
    if known_Vidya_bhushan_encoding:
        known_face_encodings.append(known_Vidya_bhushan_encoding[0])
        known_face_names.append("Vidya Bhushan")
        
    if known_Aditya_encoding:
        known_face_encodings.append(known_Aditya_encoding[0])
        known_face_names.append("Aditya")

    if known_Amit_encoding:
        known_face_encodings.append(known_Amit_encoding[0])
        known_face_names.append("Amit")

    if known_Neeraj_encoding:
        known_face_encodings.append(known_Neeraj_encoding[0])
        known_face_names.append("Neeraj")
        
    if known_pratik_encoding:
        known_face_encodings.append(known_pratik_encoding[0])
        known_face_names.append("Pratik")

    if known_Ishan_encoding:
        known_face_encodings.append(known_Ishan_encoding[0])
        known_face_names.append("Ishan")

except Exception as e:
    print(f"Error loading images or encoding faces: {e}")

# Initialize webcam
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if not ret:
        print("Failed to capture video")
        break

    # Resize frame for faster processing (optional)
    frame_small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # Find all face locations in the current frame
    face_locations = face_recognition.face_locations(frame_small)
    face_encodings = face_recognition.face_encodings(frame_small, face_locations)

    # Loop through each face found in the frame 
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Scale back the face locations since we resized the frame
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Check if the face matches any known faces 
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a box around the faces and label with the name 
        color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)  # Red for unknown, Green for known
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the resulting frame
    cv2.imshow("Video", frame)

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows 
video_capture.release()
cv2.destroyAllWindows()
