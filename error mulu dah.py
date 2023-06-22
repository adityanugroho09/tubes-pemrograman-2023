import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp
import time
import random

# Inisialisasi Mediapipe hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def track_fingers():
    # Webcamnya
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
            # Seting Video Capturenya
            ret, img = cap2.read()
            if not ret:
                break
            img = cv2.flip(img, 1)

            # Convert img ke RGB
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Proses img pada mediapipe
            results = hands.process(image_rgb)

            # pengecekan Handlandmark terdeteksi
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Membuat variabel pada finger landmark
                    index_finger_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    index_x, index_y = int(index_finger_landmark.x * img.shape[1]), int(index_finger_landmark.y * img.shape[0])
                    cv2.circle(img, (index_x, index_y), 8, (255, 255, 255), -1)

                    print(index_x, index_y)
                    # kondisi ketika jari menyentuk koordinat yang ditentukan di menu GBK
                    if 400 <= index_x <= 550 and img.shape[0] // 2 - 20 <= index_y <= img.shape[0] // 2 + 20:
                        selected_menu = "GBK"
                         
                    # kondisi ketika jari menyentuk koordinat yang ditentukan di menu GOS
                    elif 40 <= index_x <= 230 and img.shape[0] // 2 - 20 <= index_y <= img.shape[0] // 2 + 20:
                        selected_menu = "GOS"
                    elif selected_menu == "GBK":
                        selected_menu = "GBK"
                    elif selected_menu == "GOS":
                        selected_menu = "GOS"
            #Parameter = (x,y), font,warna,
            cv2.putText(img, "SUIT GAMES",  (180, img.shape[0] // 2 - 150), cv2.FONT_HERSHEY_DUPLEX, 1, (80, 127, 255), thickness=4)
            cv2.putText(img, "Gunakan Telunjuk Untuk Memilih Game", (1, img.shape[0] // 2 - 100), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))
            cv2.putText(img, "Tekan 's' Untuk Keluar :(",  (80, img.shape[0] // 2 + 150), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))
            cv2.putText(img, "Tekan 'Backspace' Buat Balikan",  (80, img.shape[0] // 2 + 180), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))

            #KOTAK & TEXT
            if selected_menu == None:
                cv2.rectangle(img, (20, img.shape[0] // 2 - 50), (250, img.shape[0] // 2 + 50), (0, 140, 255), -1)
                cv2.rectangle(img, (380,img.shape[0] // 2 - 50), (780,img.shape[0] // 2 + 50), (0, 140, 255),  -1)
                cv2.putText(img, "Suit GBK", (400, img.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if selected_menu == "GBK" else (225, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(img, "Suit GOS", (img.shape[1] - 600, img.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if selected_menu == "GOS" else (225, 0, 0), 2, cv2.LINE_AA)

            # cek menu dengan memberikan pesan
            if selected_menu == "GBK":
                # hapus menu yang ditampilkan
                cv2.rectangle(img, (20, img.shape[0] // 2 - 50), (img.shape[1] - 20, img.shape[0] // 2 + 50), (0, 0, 0), -1)
                # menampilkan pesan memasuki menu game
                cv2.putText(img, "Selamat datang di GBK game", (img.shape[1] // 2 - 200, img.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 0, 0), 2, cv2.LINE_AA)
                # cek menu kembali
                cv2.putText(img, "Back", (img.shape[1] // 2 - 200, img.shape[0] // 2 - 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 0, 0), 2, cv2.LINE_AA)
                cv2.destroyAllWindows()
                main_game(selected_menu)

            elif selected_menu == "GOS":
                # hapus menu yang ditampilkan
                cv2.rectangle(img, (20, img.shape[0] // 2 - 50), (img.shape[1] - 20, img.shape[0] // 2 + 50), (0, 0, 0),-1)
                # menampilkan pesan memasuki menu game
                cv2.putText(img, "Selamat datang di GOS game", (img.shape[1] // 2 - 180, img.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.destroyAllWindows()
                main_game(selected_menu)

            # menampilkan posisi dari jari
            cv2.imshow('Finger Tracking', img)

            # keluar program
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break

    # menutup window 
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
        if selected_menu == "GBK":
            imgBG = cv2.imread("Resources/BG.png")
        elif selected_menu == "GOS":
            imgBG = cv2.imread("Resources/Background.png")
        
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

                    # Jika tangan terdeteksi 
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
                        elif selected_menu == "GOS":
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
                        if selected_menu == "GOS":
                            # Palyer Menang
                            if (playerMove == 4 and randomNumber == 5) \
                                or (playerMove == 5 and randomNumber == 6) \
                                    or (playerMove == 6 and randomNumber == 4):
                                scores[1] += 1
                    
                            # AI Menang
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

        # Klik 'space' untuk memulai game, 'backspace' untuk kembali, dan 's' keluar
        key = cv2.waitKey(1)
        if key == 32:
            startGame = True
            initialTime = time.time()
            stateResult = False
        elif key == 8:
            cv2.destroyAllWindows()
            track_fingers()
        elif key == ord('s'):
            break
            

if __name__ == '__main__':
    track_fingers()
    main_game()
