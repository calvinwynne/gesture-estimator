import cv2
import os
import time
import numpy as np

FILE_NAME = "Sample"
DURATION = 5
explicit = [
    "Slapping",
    "Ducking",
    "PhoneOut",
    "ReachingOut",
    "Punching",
    "Shaking hands",
    "Bowing",
    "handsUp"]

implicit = [
    "Walking",
    "Yawning",
    "Defending",
    "Standing",
    "Sitting",
    "ScratchingHead",
    "Jumping",
    "Limping"]

FPS = 30
FOURCC = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
RESOLUTION = (1280, 720)

def record(action):
    file_name = os.path.join("/Users/calvin/Documents/NUIG/Thesis/dataset", FILE_NAME, action + ".mp4")
    if not os.path.isdir(os.path.join("/Users/calvin/Documents/NUIG/Thesis/dataset", FILE_NAME)):
        os.mkdir(os.path.join("/Users/calvin/Documents/NUIG/Thesis/dataset", FILE_NAME))
    
    start = time.perf_counter()
    writer = cv2.VideoWriter(file_name, FOURCC, FPS, RESOLUTION)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring emtpty camera frame.")
            continue
        writer.write(image)
        timer = int(time.perf_counter() - start)
        image = cv2.putText(image, str(timer), (image.shape[1]-70, image.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.imshow("Capture", image)
        if (cv2.waitKey(1) & 0xFF) == ord("q") or timer >= DURATION:
            break
    cap.release()
    cv2.destroyAllWindows()
    
    if input(action + ": Retry?") in ["y", "Y"]:
        os.remove(file_name)
        record(action)
    

for action in explicit + implicit:
    record(action)