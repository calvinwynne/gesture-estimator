import os 
import cv2
import time
from humanPoseDetector import poseDetector

class Extractor:
    def __init__(self, render=False, scaleFactor=1):
        self.actions   = ['handshake', 'waving', 'yawning', 'walking', 'bowing', 'punching', 'standing', 'sitting', 'touchinghead', 'defending', 'reachingup']
        self.parentDir = os.path.join("/", *"/Users/calvin/Documents/NUIG/Thesis/".split("/"))
        self.detector  = poseDetector(renderOutput=render, scaleFactor=scaleFactor)

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
                    outputFile = os.path.join(self.parentDir, "extractedData", action, video_file.split(".")[0] + ".npy")
                    if not os.path.isfile(outputFile):
                        print("\t", video_file, " "*(20-len(video_file)), end="")
                        self.detector.run(os.path.join(action_folder, video_file))
                        self.detector.save(outputFile)
                        print(" Duration : {}".format(round(time.perf_counter() - start, 3)))
                    else:
                        print("\t", video_file, " "*(20-len(video_file)), "exists")
        print("\nTotal Execution time:", round(time.perf_counter() - master_timer))
        
    def runSample(self, file_path):
        self.detector.run(file_path)
        
                    
ex = Extractor(render=False, scaleFactor=0.5)
ex.run()