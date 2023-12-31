import cv2
import mediapipe as mp

# Initialize Mediapipe hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def track_fingers():
    # Open video capture
    cap2 = cv2.VideoCapture(0)

    # Initialize hands object
    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:

        # Initialize selected menu variable
        selected_menu = None

        while cap2.isOpened():
            # Read frame from the video capture
            ret, frame = cap2.read()
            if not ret:
                break

            # Flip the frame horizontally for a mirrored display
            frame = cv2.flip(frame, 1)

            # Convert the frame to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with Mediapipe
            results = hands.process(image_rgb)

            # Check if hand landmarks are detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get the coordinates of the index finger
                    index_finger_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    index_x, index_y = int(index_finger_landmark.x * frame.shape[1]), int(index_finger_landmark.y * frame.shape[0])

                    # Draw circle at the finger tip
                    cv2.circle(frame, (index_x, index_y), 8, (0, 255, 0), -1)

                    # Check if the forefinger is on the "English" menu
                    if 20 <= index_x <= 120 and frame.shape[0] // 2 - 50 <= index_y <= frame.shape[0] // 2 + 50:
                        selected_menu = "English"
                    # Check if the forefinger is on the "Jawa" menu
                    elif frame.shape[1] - 120 <= index_x <= frame.shape[1] - 20 and frame.shape[0] // 2 - 50 <= index_y <= frame.shape[0] // 2 + 50:
                        selected_menu = "Jawa"
                    elif selected_menu == "English":
                        selected_menu = "English"
                    elif selected_menu == "Jawa":
                        selected_menu = "Jawa"

            if selected_menu == None:
                # Add text labels
                cv2.putText(frame, "English", (20, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if selected_menu == "English" else (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, "Jawa", (frame.shape[1] - 120, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if selected_menu == "Jawa" else (255, 255, 255), 2, cv2.LINE_AA)

            # Check the selected menu and display the appropriate message
            if selected_menu == "English":
                # Erase all menu displays
                cv2.rectangle(frame, (20, frame.shape[0] // 2 - 50), (frame.shape[1] - 20, frame.shape[0] // 2 + 50), -1)
                # Display the message
                cv2.putText(frame, "You are in English Game", (frame.shape[1] // 2 - 200, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                # Display back menu
                cv2.putText(frame, "Back", (frame.shape[1] // 2 - 200, frame.shape[0] // 2 - 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            elif selected_menu == "Jawa":
                # Erase all menu displays
                cv2.rectangle(frame, (20, frame.shape[0] // 2 - 50), (frame.shape[1] - 20, frame.shape[0] // 2 + 50), -1)
                # Display the message
                cv2.putText(frame, "You are in Jawa Game", (frame.shape[1] // 2 - 180, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display the frame with the finger positions and menu selection
            cv2.imshow('Finger Tracking', frame)

            # Check for 'q' key press to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the video capture and close all windows
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    track_fingers()
