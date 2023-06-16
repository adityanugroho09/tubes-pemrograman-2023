import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp
import time
import random


class RockPaperScissorsGame:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)

        self.detector = HandDetector(maxHands=1)

        self.timer = 0
        self.stateResult = False
        self.startGame = False

        self.scores = [0, 0]  # [AI, PLAYER]

    def track_fingers(self):
        self.cap2 = cv2.VideoCapture(0)

        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands

        with mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:

            self.selected_menu = None

            while self.cap2.isOpened():
                ret, frame = self.cap2.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = hands.process(image_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        index_finger_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        index_x, index_y = int(index_finger_landmark.x * frame.shape[1]), int(
                            index_finger_landmark.y * frame.shape[0])

                        cv2.circle(frame, (index_x, index_y), 8, (0, 255, 0), -1)

                        if 20 <= index_x <= 120 and frame.shape[0] // 2 - 50 <= index_y <= frame.shape[0] // 2 + 50:
                            self.selected_menu = "English"
                        elif frame.shape[1] - 120 <= index_x <= frame.shape[1] - 20 and frame.shape[
                            0] // 2 - 50 <= index_y <= frame.shape[0] // 2 + 50:
                            self.selected_menu = "Jawa"
                        elif self.selected_menu == "English":
                            self.selected_menu = "English"
                        elif self.selected_menu == "Jawa":
                            self.selected_menu = "Jawa"

                if self.selected_menu is None:
                    cv2.putText(frame, "English", (20, frame.shape[0] // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0) if self.selected_menu == "English" else (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, "Jawa", (frame.shape[1] - 120, frame.shape[0] // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0) if self.selected_menu == "Jawa" else (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow('Finger Tracking', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.cap2.release()
        cv2.destroyAllWindows()

    def main_game(self):
        while True:
            imgBG = cv2.imread("Resources/BG.png")
            success, img = self.cap.read()

            imgScaled = cv2.resize(img, (0, 0), None, 0.875, 0.875)
            imgScaled = imgScaled[:, 80:480]

            hands, img = self.detector.findHands(imgScaled)

            if self.startGame:
                if self.stateResult is False:
                    self.timer = time.time() - self.initialTime
                    cv2.putText(imgBG, str(int(self.timer)), (200, 435), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 255), 4)

                    if self.timer > 3:
                        self.stateResult = True
                        self.timer = 0
                        if hands:
                            playerMove = None
                            hand = hands[0]
                            fingers = self.detector.fingersUp(hand)

                            if fingers == [0, 0, 0, 0, 0]:
                                playerMove = 1
                            if fingers == [1, 1, 1, 1, 1]:
                                playerMove = 2
                            if fingers == [0, 1, 1, 0, 0]:
                                playerMove = 3

                            randomNumber = random.randint(1, 3)
                            imgAI = cv2.imread(f'Resources/{randomNumber}.png', cv2.IMREAD_UNCHANGED)
                            imgBG = cvzone.overlayPNG(imgBG, imgAI, (200, 200))

                            if (playerMove == 1 and randomNumber == 3) \
                                    or (playerMove == 2 and randomNumber == 1) \
                                    or (playerMove == 3 and randomNumber == 2):
                                self.scores[1] += 1

                            if (playerMove == 3 and randomNumber == 1) \
                                    or (playerMove == 1 and randomNumber == 2) \
                                    or (playerMove == 2 and randomNumber == 3):
                                self.scores[0] += 1

                            print(playerMove)

            imgBG[233:653, 414:1195] = imgScaled

            if self.stateResult:
                imgBG = cvzone.overlayPNG(imgBG, imgAI, (200, 200))

            cv2.putText(imgBG, str(self.scores[0]), (200, 100), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 255), 4)
            cv2.putText(imgBG, str(self.scores[1]), (435, 100), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 255), 4)

            cv2.imshow("imgBG", imgBG)

            key = cv2.waitKey(1)
            if key == ord('s'):
                self.startGame = True
                self.initialTime = time.time()
                self.stateResult = False

    def run(self):
        self.track_fingers()
        self.main_game()


if __name__ == '__main__':
    game = RockPaperScissorsGame()
    game.run()
