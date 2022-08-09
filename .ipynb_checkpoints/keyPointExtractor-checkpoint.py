import math
import numpy as np
import cv2
import time
import sys, os

import mediapipe as mp
import pandas as pd
import pprint
from threading import Thread
from numba import njit

_empty = None

class CustomThread(Thread):
    def __init__(self, func, args):
        Thread.__init__(self)
        self.func = func
        self.args = args
 
    def run(self):
        self.ret = self.func(*self.args)

        
class keyPointExtractor:
    def __init__(self, renderOutput=True, scaleFactor=1):
        
        self.RENDER_OUTPUT = renderOutput
        self.SCALE_FACTOR = scaleFactor
        self.COLUMN_NAMES = ["x", "y", "z", "v", "xn", "yn", 'a0', 'd0', 'mx0', 'my0', 'a15', 'd15', 'mx15', 'my15', 'a16', 'd16', 'mx16', 'my16', 'a33', 'd33', 'mx33', 'my33']
        self.mp_pose = mp.solutions.pose
        self.mp_pose.MODEL_COMPLEXITY = 2
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.keyPoints = None
        self.dataSeries = []
    
    def vstack_image(self, images=[]):
        out = images[0].astype(np.uint8)
        for image in images[1:]:
            out = cv2.vconcat((out, image.astype(np.uint8)))
        return out

    def hstack_image(self, images=[]):
        out = images[0].astype(np.uint8)
        for image in images[1:]:
            out = cv2.hconcat((out, image.astype(np.uint8)))
        return out

    def resize_image(self, image, fixed=False):
        return cv2.resize(image, (int(image.shape[1] * self.SCALE_FACTOR), int(image.shape[0] * self.SCALE_FACTOR)), interpolation=cv2.INTER_AREA)

    def drawStencil(self, imageDimension):
        image = np.zeros(imageDimension)
        image = self.drawCircles(image, self.keyPoints, np.arange(33))
        return image

    def drawCircles(self, image, keyPoints, circle_points, columns=(4, 6), small=False):
        _kp = keyPoints[:, columns[0]:columns[1]+1].astype("int32")
        for point in circle_points:
            _temp = _kp[point, :]
            if small:
                image = cv2.circle(image, _temp, radius=1, color=(255, 255, 255), thickness=2)     
            else:
                image = cv2.circle(image, _temp, radius=5, color=(0, 0, 255), thickness=-10)
                image = cv2.circle(image, _temp, radius=7, color=(255, 255, 255), thickness=2)   
        return image

    def annotateMeasurements(self, image, measurements):
        for measure in measurements:
            if measure.get("midpoint", None):
                x, y = measure.get("midpoint")
                image = cv2.putText(image, str(measure.get("distance", "null")), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=1, thickness=2, color=(255, 255, 255), lineType=cv2.LINE_AA)
        return image
    
    def storeKeyPoints(self, results, shape):
        keyPoints = np.zeros((34, 6))
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            keyPoints[idx, 0] = lm.x 
            keyPoints[idx, 1] = lm.y
            keyPoints[idx, 2] = round(lm.z, 3)
            keyPoints[idx, 3] = round(lm.visibility, 3)
        keyPoints[33, 0] = (keyPoints[24, 0] + keyPoints[23, 0]) / 2
        keyPoints[33, 1] = (keyPoints[24, 1] + keyPoints[23, 1]) / 2

        keyPoints[:, 4] = keyPoints[:, 0] * shape[1]
        keyPoints[:, 5] = keyPoints[:, 1] * shape[0]
        #print(pd.DataFrame(keyPoints, columns=["x", "y", "z", "v", "xn", "yn"]))
        return keyPoints

   
    def drawLinesWithReference(self, image, referencePoint, pointList, pointListColumns):
        '''
        referencePoint   -> idx
        pointList        -> [ ... ]
        pointListColumns -> (x_col, y_col)
        '''
        _kp = self.keyPoints[pointList, pointListColumns[0]:pointListColumns[1]+1].astype("int32")
        referencePoint = self.keyPoints[referencePoint, 4:6].astype("int32")
        for _pt in _kp:
            image = cv2.line(image, referencePoint, _pt, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        return image

    def find_angleDistance(self, _keyPoints, referencePoint, pointList, pointName):
        attach = np.ones((34, 4), dtype="int32")
        for point in pointList:
            y2y1 = max(_keyPoints[referencePoint, 5] - _keyPoints[point, 5], 1)
            x2x1 = max(_keyPoints[referencePoint, 4] - _keyPoints[point, 4], 1)
            attach[point, 0] = math.degrees(math.atan(y2y1/x2x1))
            attach[point, 1] = (y2y1**2) + (x2x1**2)              # sqrt this
            attach[point, 2] = (_keyPoints[referencePoint, 4] + _keyPoints[point, 4])
            attach[point, 3] = (_keyPoints[referencePoint, 5] + _keyPoints[point, 5])
        attach[:, 1] = np.sqrt(attach[:, 1])
        attach[:, 2] = attach[:, 2] / 2
        attach[:, 3] = attach[:, 3] / 2
        #_keyPoints = np.concatenate((_keyPoints, attach), axis=1)
        return attach

    def derieveMap(self, stencil, kp):
        _keyPoint = self.find_angleDistance(self.keyPoints, kp, np.arange(33), str(kp))
        if self.RENDER_OUTPUT :
            stencilN  = self.drawLinesWithReference(stencil, kp, np.arange(33), (4, 5))
        else:
            stencilN = None
        return stencilN, _keyPoint


    def drawSingleLines(self, image, _keyPoints, pointList):
        _kp = _keyPoints[:, 4:6].astype("int32")
        for pt1, pt2 in pointList:
            image = cv2.line(image, _kp[pt1, :], _kp[pt2, :], color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        return image
    
    def collectDataSeries(self):
        self.dataSeries.append(self.keyPoints)
        
    def save(self, filePath):
        np.save(filePath, self.dataSeries)
        
    def renderFrame(self, frameNumber=0):
        return pd.DataFrame(self.dataSeries[frameNumber], columns=self.COLUMN_NAMES)

    def extractKeyPoints(self, image, results):

        stencil1 = np.zeros(image.shape)
        self.mp_drawing.draw_landmarks(
                        stencil1,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

        if self.RENDER_OUTPUT:    
            stencil = self.drawStencil(image.shape)
        else:
            stencil = np.zeros((0, 0))

        self.derieveMap(stencil.copy(),  0)
        sthread2 = CustomThread(self.derieveMap, (stencil.copy(),  0))
        sthread3 = CustomThread(self.derieveMap, (stencil.copy(), 15))
        sthread4 = CustomThread(self.derieveMap, (stencil.copy(), 16))
        sthread5 = CustomThread(self.derieveMap, (stencil.copy(), 33))    

        sthread2.start()
        sthread3.start()
        sthread4.start()
        sthread5.start()

        if self.RENDER_OUTPUT:
            vstack2  = self.vstack_image([image, stencil1])

        sthread2.join()
        sthread3.join()
        stencil2, kpoints2 = sthread2.ret
        stencil3, kpoints3 = sthread3.ret

        if self.RENDER_OUTPUT:
            vstack1 = self.vstack_image([stencil3, stencil2]) 

        sthread4.join()
        sthread5.join()

        stencil4, kpoints4 = sthread4.ret
        stencil5, kpoints5 = sthread5.ret

        if self.RENDER_OUTPUT:
            vstack3 = self.vstack_image([stencil4, stencil5])
            hstack1 = self.hstack_image([vstack1, vstack2, vstack3])
        else:
            hstack1 = image

        kpoints = np.concatenate((self.keyPoints, kpoints2, kpoints3, kpoints4, kpoints5), axis=1)
        return hstack1, kpoints 
    
    def run(self, videoFile):
        cap = cv2.VideoCapture(videoFile)
        self.keyPoints  = None
        self.dataSeries = []
        with self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
            pose.enable_segmentation = False
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring emtpty camera frame.")
                    break

                image = self.resize_image(image)
                imageShape = np.array(image.shape)

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                if self.RENDER_OUTPUT:
                    image.flags.writeable = True
                    self.mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

                self.keyPoints = self.storeKeyPoints(results, imageShape)
                image, self.keyPoints = self.extractKeyPoints(image, results)
                self.collectDataSeries()
                if self.RENDER_OUTPUT:
                    cv2.imshow("stack", cv2.flip(image, 1))
                    if cv2.waitKey(1) & (0xFF == 27 | 0xFF == ord("q")):
                        cv2.destroyAllWindows()
                        break
            cap.release()

start = time.perf_counter()
executor = keyPointExtractor(renderOutput=False, scaleFactor=0.5)
executor.run("/Users/calvin/Documents/NUIG/Thesis/Movies/bowing/bowing8.MP4")
executor.save("/Users/calvin/Documents/NUIG/Thesis/Movies/bowing/bowing8")
executor.renderSample()
print("Duration   : {}".format(time.perf_counter() - start))