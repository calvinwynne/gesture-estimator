import os 
import cv2
import time
from humanPoseDetector import poseDetector


class Extractor:
    def __init__(self, render=False):
        self.actions   = ['handshake', 'waving', 'yawning', 'walking', 'bowing', 'punching', 'standing',
                          'sitting', 'scratchingHead', 'defending', 'reachingUp']
        self.parentDir = os.path.join("/", *"/Users/calvin/Documents/NUIG/Thesis/".split("/"))
        self.detector  = poseDetector(renderOutput=render, scaleFactor=0.5)

    def run(self):
        master_timer = time.perf_counter()
        if not os.path.isdir(os.path.join(self.parentDir, "extractedData")):
            os.mkdir(os.path.join(self.parentDir, "extractedData"))

        for action in self.actions:
            action_folder = os.path.join(self.parentDir, "Movies", action)
            print(action.title())
            for video_file in os.listdir(action_folder):
                if video_file.split(".")[-1].lower() == "mp4":
                    if not os.path.isdir(os.path.join(self.parentDir, "extractedData", action)):
                        os.mkdir(os.path.join(self.parentDir, "extractedData", action))
                    start = time.perf_counter()
                    outputFile = os.path.join(self.parentDir, "extractedData", action, video_file.split(".")[0])
                    if not os.path.isfile(outputFile+".npy"):
                        print("\t", video_file, " "*(20-len(video_file)), end="")
                        self.detector.run(os.path.join(action_folder, video_file))
                        self.detector.save(outputFile)
                        print(" Duration : {}".format(round(time.perf_counter() - start, 3)))
                    else:
                        print("\t", video_file, " "*(20-len(video_file)), "exists")
        print("\nTotal Execution time:", round(time.perf_counter() - master_timer))
                    
ex = Extractor(render=False)
ex.run()