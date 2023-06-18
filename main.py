import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp
import time
import random

# Initialize Mediapipe hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def track_fingers():
    # Open video capture
    cap2 = cv2.VideoCapture(0)
    cap2.set(3,1280)
    cap2.set(4,720)

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

                    print(index_x, index_y)
                    # Check if the forefinger is on the "GBK" menu
                    if 320 <= index_x <= 450 and frame.shape[0] // 2 - 20 <= index_y <= frame.shape[0] // 2 + 20:
                        selected_menu = "GBK"
                    # Check if the forefinger is on the "Normal" menu
                    elif 780 <= index_x <= 975 and frame.shape[0] // 2 - 20 <= index_y <= frame.shape[0] // 2 + 20:
                        selected_menu = "Normal"
                    elif selected_menu == "GBK":
                        selected_menu = "GBK"
                    elif selected_menu == "Normal":
                        selected_menu = "Normal"

            cv2.putText(frame, "Gunakan Telunjuk Untuk Memilih", (frame.shape[0] // 2, frame.shape[0] // 2 - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
            cv2.putText(frame, "Tekan 'Q' Untuk Keluar Dari Game", (frame.shape[0] // 2, frame.shape[0] // 2 + 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

            if selected_menu == None:
                # Add text labels
                cv2.rectangle(frame, (20, frame.shape[0] // 2 - 50), (150, frame.shape[0] // 2 + 50), (0, 0, 0), -1)
                cv2.putText(frame, "Suit GBK", (320, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if selected_menu == "GBK" else (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, "Suit Normal", (frame.shape[1] - 500, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if selected_menu == "Normal" else (255, 255, 255), 2, cv2.LINE_AA)

            # Check the selected menu and display the appropriate message
            if selected_menu == "GBK":
                # Erase all menu displays
                cv2.rectangle(frame, (20, frame.shape[0] // 2 - 50), (frame.shape[1] - 20, frame.shape[0] // 2 + 50), (0, 0, 0), -1)
                # Display the message
                cv2.putText(frame, "You are in GBK Game", (frame.shape[1] // 2 - 200, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                # Display back menu
                cv2.putText(frame, "Back", (frame.shape[1] // 2 - 200, frame.shape[0] // 2 - 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.destroyAllWindows()
                main_game(selected_menu)

            elif selected_menu == "Normal":
                # Erase all menu displays
                cv2.rectangle(frame, (20, frame.shape[0] // 2 - 50), (frame.shape[1] - 20, frame.shape[0] // 2 + 50), (0, 0, 0),-1)
                # Display the message
                cv2.putText(frame, "You are in Normal Game", (frame.shape[1] // 2 - 180, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.destroyAllWindows()
                main_game(selected_menu)

            # Display the frame with the finger positions and menu selection
            cv2.imshow('Finger Tracking', frame)

            # Check for 'q' key press to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the video capture and close all windows
    cap2.release()
    cv2.destroyAllWindows()

def main_game(selected_menu):
    # Penggunaan Camera
    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)

    detector = HandDetector(maxHands=1)

    # Timer
    timer = 0
    stateResult = False
    startGame = False

    # Array Score
    scores = [0,0] # [AI, PLAYER]

    selected_menu = selected_menu

    while True:
        imgBG = cv2.imread("Resources/BG.png")
        success, img = cap.read()

        # Mengatur Ukuran Layar Kamera
        imgScaled = cv2.resize(img,(0,0),None,0.875,0.875)
        imgScaled = imgScaled[:,80:480]

        # Find Hands
        hands, img = detector.findHands(imgScaled)

        # Game Dimulai
        if startGame:

            if stateResult is False:
                timer = time.time() - initialTime
                cv2.putText(imgBG, str(int(timer)), (605, 435), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 255), 4)

                # Timer
                if timer > 3:
                    stateResult = True
                    timer = 0

                    # Jika tangan terdeteksi maka :
                    if hands:
                        playerMove = None
                        hand = hands[0]
                        fingers = detector.fingersUp(hand)

                        # Deteksi Jari dan Penentuan Palyer Move
                        if selected_menu == "GBK":
                            if fingers == [0,0,0,0,0]:
                                playerMove = 1
                            if fingers == [1,1,1,1,1]:
                                playerMove = 2
                            if fingers == [0,1,1,0,0]:
                                playerMove = 3
                            # Generate Nomor Acak
                            randomNumber = random.randint(1, 3)
                        elif selected_menu == "Normal":
                            if fingers == [1,0,0,0,0]:
                                playerMove = 4
                            if fingers == [0,1,0,0,0]:
                                playerMove = 5
                            if fingers == [0,0,0,0,1]:
                                playerMove = 6
                            randomNumber = random.randint(4, 6)

                        # Akses Gambar
                        imgAI = cv2.imread(f'Resources/{randomNumber}.png',cv2.IMREAD_UNCHANGED)
                        imgBG = cvzone.overlayPNG(imgBG, imgAI, (149, 310))

                        if selected_menu == "GBK":
                            # Kondisi Player Menang
                            if (playerMove == 1 and randomNumber == 3) \
                                or (playerMove == 2 and randomNumber == 1) \
                                    or (playerMove == 3 and randomNumber == 2):
                                scores[1] += 1
                        
                            # Kondisi AI Menang
                            if (playerMove == 3 and randomNumber == 1) \
                                or (playerMove == 1 and randomNumber == 2) \
                                    or (playerMove == 2 and randomNumber == 3):
                                scores[0] += 1
                            print(playerMove)
                        if selected_menu == "Normal":
                            # Palyer Wins
                            if (playerMove == 4 and randomNumber == 5) \
                                or (playerMove == 5 and randomNumber == 6) \
                                    or (playerMove == 6 and randomNumber == 4):
                                scores[1] += 1
                    
                            # AI Wins
                            if (playerMove == 5 and randomNumber == 4) \
                                or (playerMove == 6 and randomNumber == 5) \
                                    or (playerMove == 4 and randomNumber == 6):
                                scores[0] += 1

        #  Atur Ukuran Display Camera
        imgBG[234:654, 795:1195] = imgScaled

        # Tata Letak Gambar AI
        if stateResult:
            imgBG = cvzone.overlayPNG(imgBG, imgAI, (149, 310))

        # Tata Letak Score
        cv2.putText(imgBG, str(scores[0]), (410, 215), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 6)
        cv2.putText(imgBG, str(scores[1]), (1112, 215), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 6)

        # Menampilkan tampilan layar
        cv2.imshow("imgBG", imgBG) 

        # Klik S Untuk Memulai Game
        key = cv2.waitKey(1)
        if key == ord('s'):
            startGame = True
            initialTime = time.time()
            stateResult = False
        elif key == ord('b'):
            cv2.destroyAllWindows()
            track_fingers()
        elif key == ord('q'):
            break
            

if __name__ == '__main__':
    track_fingers()
    main_game()