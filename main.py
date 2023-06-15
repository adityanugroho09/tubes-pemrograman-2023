import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp
import time
import random

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

detector = HandDetector(maxHands=1)

timer = 0
stateResult = False
startGame = False

scores = [0,0] # [AI, PLAYER]

while True:
    imgBG = cv2.imread("Resources/BG.png")
    success, img = cap.read()

    imgScaled = cv2.resize(img,(0,0),None,0.875,0.875)
    imgScaled = imgScaled[:,80:480]

    # Find Hands
    hands, img = detector.findHands(imgScaled)

    if startGame:

        if stateResult is False:
            timer = time.time() - initialTime
            cv2.putText(imgBG, str(int(timer)), (200, 435), cv2.FONT_HERSHEY_PLAIN, 6, (255,0,255),4)

            if timer > 3:
                stateResult = True
                timer = 0
                if hands:
                    playerMove = None
                    hand = hands[0]
                    fingers = detector.fingersUp(hand)

                    if fingers == [0,0,0,0,0]:
                        playerMove = 1
                    if fingers == [1,1,1,1,1]:
                        playerMove = 2
                    if fingers == [0,1,1,0,0]:
                        playerMove = 3

                    randomNumber = random.randint(1, 3)
                    imgAI = cv2.imread(f'Resources/{randomNumber}.png',cv2.IMREAD_UNCHANGED)
                    imgBG = cvzone.overlayPNG(imgBG, imgAI, (200,200))

                    # Palyer Wins
                    if (playerMove == 1 and randomNumber == 3) \
                        or (playerMove == 2 and randomNumber == 1) \
                            or (playerMove == 3 and randomNumber == 2):
                        scores[1] += 1
                    
                    # AI Wins
                    if (playerMove == 3 and randomNumber == 1) \
                        or (playerMove == 1 and randomNumber == 2) \
                            or (playerMove == 2 and randomNumber == 3):
                        scores[0] += 1

                    print(playerMove)

    imgBG[233:653,414:1195] = imgScaled

    if stateResult:
        imgBG = cvzone.overlayPNG(imgBG, imgAI, (200,200)) # Gambar AI

    cv2.putText(imgBG, str(scores[0]), (200, 100), cv2.FONT_HERSHEY_PLAIN, 6, (255,0,255),4) # AI Score
    cv2.putText(imgBG, str(scores[1]), (435, 100), cv2.FONT_HERSHEY_PLAIN, 6, (255,0,255),4) # Palyer Score

    # cv2.imshow("Image", img)
    cv2.imshow("imgBG", imgBG) # Menampilkan tampilan layar
    # cv2.imshow("Scaled", imgScaled)

    # Klik S Untuk Memulai Game
    key = cv2.waitKey(1)
    if key == ord('s'):
        startGame = True
        initialTime = time.time()
        stateResult = False