import cv2
import mediapipe as mp
import time
import controller as cnt

# Allow some time for camera to warm up
time.sleep(2.0)

# Initialize MediaPipe components
mp_draw = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands

# Finger tip IDs
tipIds = [4, 8, 12, 16, 20]

# Open video capture
video = cv2.VideoCapture(0)


# Define a function to display the count and LED status
def display_count(image, total):
    # Create a filled rectangle for display
    cv2.rectangle(image, (20, 300), (270, 425), (0, 255, 0), cv2.FILLED)
    # Display the total count
    cv2.putText(image, str(total), (45, 375), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
    cv2.putText(image, "LED", (100, 375), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)


with mp_hand.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        ret, image = video.read()
        if not ret:
            break

        # Process the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        lmList = []
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                for id, lm in enumerate(hand_landmark.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                mp_draw.draw_landmarks(image, hand_landmark, mp_hand.HAND_CONNECTIONS)

        # Count fingers
        fingers = []
        if lmList:
            fingers.append(1 if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1] else 0)
            for id in range(1, 5):
                fingers.append(1 if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2] else 0)

            total = fingers.count(1)
            cnt.led(total)  # Control LEDs based on finger count
            display_count(image, total)  # Display the total count on the frame

        # Show the image
        cv2.imshow("Frame", image)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

# Release resources
video.release()
cv2.destroyAllWindows()
