import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp
import time
import random


class FingerGame:
    def __init__(self):
        # Initialize Mediapipe hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

        # Open video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)

        # Initialize hands object
        self.detector = HandDetector(maxHands=1)

        # Initialize selected menu variable
        self.selected_menu = None

        # Initialize game variables
        self.timer = 0
        self.state_result = False
        self.start_game = False
        self.scores = [0, 0]  # [AI, PLAYER]

    def run(self):
        with self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
        ) as hands:
            while self.cap.isOpened():
                # Read frame from the video capture
                ret, frame = self.cap.read()
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
                        index_finger_landmark = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        index_x, index_y = int(index_finger_landmark.x * frame.shape[1]), int(
                            index_finger_landmark.y * frame.shape[0])

                        # Draw circle at the finger tip
                        cv2.circle(frame, (index_x, index_y), 8, (0, 255, 0), -1)

                        print(index_x, index_y)
                        # Check if the forefinger is on the "GBK" menu
                        if 320 <= index_x <= 450 and frame.shape[0] // 2 - 20 <= index_y <= frame.shape[0] // 2 + 20:
                            self.selected_menu = "GBK"
                        # Check if the forefinger is on the "Normal" menu
                        elif 780 <= index_x <= 975 and frame.shape[0] // 2 - 20 <= index_y <= frame.shape[0] // 2 + 20:
                            self.selected_menu = "Normal"
                        elif self.selected_menu == "GBK":
                            self.selected_menu = "GBK"
                        elif self.selected_menu == "Normal":
                            self.selected_menu = "Normal"

                cv2.putText(frame, "Gunakan Telunjuk Untuk Memilih",
                            (frame.shape[0] // 2, frame.shape[0] // 2 - 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255))
                cv2.putText(frame, "Tekan 'Q' Untuk Keluar Dari Game",
                            (frame.shape[0] // 2, frame.shape[0] // 2 + 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255))

                if self.selected_menu is None:
                    # Add text labels
                    cv2.rectangle(frame, (20, frame.shape[0] // 2 - 50), (150, frame.shape[0] // 2 + 50), (0, 0, 0), -1)
                    cv2.putText(frame, "Suit GBK", (320, frame.shape[0] // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0) if self.selected_menu == "GBK" else (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, "Suit Normal", (frame.shape[1] - 500, frame.shape[0] // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0) if self.selected_menu == "Normal" else (255, 255, 255), 2, cv2.LINE_AA)

                # Check the selected menu and display the appropriate message
                if self.selected_menu == "GBK":
                    # Erase all menu displays
                    cv2.rectangle(frame, (20, frame.shape[0] // 2 - 50),
                                  (frame.shape[1] - 20, frame.shape[0] // 2 + 50), (0, 0, 0), -1)
                    # Display the message
                    cv2.putText(frame, "You are in GBK Game", (frame.shape[1] // 2 - 200, frame.shape[0] // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    # Display back menu
                    cv2.putText(frame, "Back", (frame.shape[1] // 2 - 200, frame.shape[0] // 2 - 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.destroyAllWindows()
                    self.main_game()

                elif self.selected_menu == "Normal":
                    # Erase all menu displays
                    cv2.rectangle(frame, (20, frame.shape[0] // 2 - 50),
                                  (frame.shape[1] - 20, frame.shape[0] // 2 + 50), (0, 0, 0), -1)
                    # Display the message
                    cv2.putText(frame, "You are in Normal Game", (frame.shape[1] // 2 - 180, frame.shape[0] // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.destroyAllWindows()
                    self.main_game()

                # Display the frame with the finger positions and menu selection
                cv2.imshow('Finger Tracking', frame)

                # Check for 'q' key press to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Release the video capture and close all windows
        self.cap.release()
        cv2.destroyAllWindows()

    def main_game(self):
        while True:
            img_bg = cv2.imread("Resources/BG.png")
            success, img = self.cap.read()

            # Resize camera frame
            img_scaled = cv2.resize(img, (0, 0), None, 0.875, 0.875)
            img_scaled = img_scaled[:, 80:480]

            # Find hands
            hands = self.detector.findHands(img_scaled)

            # Game starts
            if self.start_game:

                if self.state_result is False:
                    self.timer = time.time() - self.initial_time
                    cv2.putText(img_bg, str(int(self.timer)), (605, 435), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 255), 4)

                    # Timer
                    if self.timer > 3:
                        self.state_result = True
                        self.timer = 0

                        # If hand is detected
                        if hands:
                            player_move = None
                            hand = hands[0]
                            fingers = self.detector.fingers_up(hand)

                            # Finger detection and player move determination
                            if self.selected_menu == "GBK":
                                if fingers == [0, 0, 0, 0, 0]:
                                    player_move = 1
                                if fingers == [1, 1, 1, 1, 1]:
                                    player_move = 2
                                if fingers == [0, 1, 1, 0, 0]:
                                    player_move = 3
                                # Generate random number
                                random_number = random.randint(1, 3)
                            elif self.selected_menu == "Normal":
                                if fingers == [1, 0, 0, 0, 0]:
                                    player_move = 4
                                if fingers == [0, 1, 0, 0, 0]:
                                    player_move = 5
                                if fingers == [0, 0, 0, 0, 1]:
                                    player_move = 6
                                random_number = random.randint(4, 6)

                            # Access image
                            img_ai = cv2.imread(f'Resources/{random_number}.png', cv2.IMREAD_UNCHANGED)
                            img_bg = cvzone.overlay_png(img_bg, img_ai, (149, 310))

                            if self.selected_menu == "GBK":
                                # Player wins condition
                                if (player_move == 1 and random_number == 3) \
                                        or (player_move == 2 and random_number == 1) \
                                        or (player_move == 3 and random_number == 2):
                                    self.scores[1] += 1

                                # AI wins condition
                                if (player_move == 3 and random_number == 1) \
                                        or (player_move == 1 and random_number == 2) \
                                        or (player_move == 2 and random_number == 3):
                                    self.scores[0] += 1
                                print(player_move)
                            if self.selected_menu == "Normal":
                                # Player wins condition
                                if (player_move == 4 and random_number == 5) \
                                        or (player_move == 5 and random_number == 6) \
                                        or (player_move == 6 and random_number == 4):
                                    self.scores[1] += 1

                                # AI wins condition
                                if (player_move == 5 and random_number == 4) \
                                        or (player_move == 6 and random_number == 5) \
                                        or (player_move == 4 and random_number == 6):
                                    self.scores[0] += 1

            # Set camera display size
            img_bg[234:654, 795:1195] = img_scaled

            # AI image layout
            if self.state_result:
                img_bg = cvzone.overlay_png(img_bg, img_ai, (149, 310))

            # Score layout
            cv2.putText(img_bg, str(self.scores[0]), (410, 215), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 10)
            cv2.putText(img_bg, str(self.scores[1]), (905, 215), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 10)

            # Hand landmark points overlay
            self.mp_drawing.draw_landmarks(img_bg, hand, self.mp_hands.HAND_CONNECTIONS)

            # Play menu layout
            cv2.putText(img_bg, "Tekan 'R' Untuk Restart", (470, 670), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 4)
            cv2.putText(img_bg, "Tekan 'Q' Untuk Keluar Dari Game", (220, 730), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 4)

            # Check for 'r' key press to restart
            if cv2.waitKey(1) & 0xFF == ord('r'):
                self.start_game = False
                self.state_result = False
                self.scores = [0, 0]

            # Check for 'q' key press to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Display the frame with the game layout
            cv2.imshow("Finger Game", img_bg)

        # Release the video capture and close all windows
        self.cap.release()
        cv2.destroyAllWindows()


# Create an instance of the FingerGame class and run the game
game = FingerGame()
game.run()
