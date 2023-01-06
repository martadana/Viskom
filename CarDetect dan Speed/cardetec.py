import cv2
import numpy as np
import time
import math

class Detection():
    def __init__(self, frame, subtraction, vehicle_width_min=80, vehicle_height_min=80):
        self.__vehicle_width_min = vehicle_width_min
        self.__vehicle_height_min = vehicle_height_min
        self.__detect = []
        self.__frame = frame
        self.__subtraction = subtraction

    def subtract(self):
        self.__gray = cv2.cvtColor(self.__frame, cv2.COLOR_BGR2GRAY)
        self.__blur = cv2.GaussianBlur(self.__gray, (3,3), 5)
        self.__img_sub = self.__subtraction.apply(self.__blur)
        self.__ret, self.__img_sub = cv2.threshold(self.__img_sub, 200, 255, cv2.THRESH_BINARY)

        self.__dilat = cv2.dilate(self.__img_sub, np.ones((5,5), np.uint8))
        self.__kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        self.__expand = cv2.morphologyEx(self.__dilat, cv2.MORPH_CLOSE, self.__kernel)
        self.__expand = cv2.morphologyEx(self.__expand, cv2.MORPH_CLOSE, self.__kernel)
        self.__contour, self.__h = cv2.findContours(self.__expand, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    def validateOutline(self, contour):
        self.__x, self.__y, self.__w, self.__h = cv2.boundingRect(contour)
        self.__validate_outline = (self.__w >= self.__vehicle_width_min) and (self.__h >= self.__vehicle_height_min)

    def drawRectangle(self):
        cv2.rectangle(self.__frame, (self.__x, self.__y), (self.__x+self.__w, self.__y+self.__h), (0,255,0), 2)
    
    def drawDotCenter(self):
        self.__center_x = self.__x + int(self.__w / 2)
        self.__center_y = self.__y + int(self.__h / 2)
        self.__detect.append((self.__center_x, self.__center_y))
        cv2.circle(self.__frame, (self.__center_x, self.__center_y), 4, (0, 0,255), -1)
    
    def removeDetect(self, center):
        self.__detect.remove(center)

    def getContour(self):
        return self.__contour
    
    def getBoundingRect(self):
        return self.__x, self.__y, self.__w, self.__h
    
    def getValidateOutline(self):
        return self.__validate_outline
    
    def getDetection(self):
        return self.__detect

class Counter():
    def __init__(self, line_height=450, offset=6):
        self.__left_vehicle = 0
        self.__right_vehicle = 0
        self.__line_height = line_height
        self.__offset = offset
    
    def drawLine(self, frame, left, right, color):
        cv2.line(frame, (left, self.__line_height), (right, self.__line_height), color, 5)
        
    def detecLeft(self, left, right, x, y):
        if y<(self.__line_height+self.__offset) and y>(self.__line_height-self.__offset) and x>=left and x<=right:
            self.__left_vehicle+=1
            return True
        else:
            return False
    
    def detecRight(self, left, right, x, y):
        if y<(self.__line_height+self.__offset) and y>(self.__line_height-self.__offset) and x>=left and x<=right:
            self.__right_vehicle+=1
            return True
        else:
            return False
    
    def putLeftVehicleCounter(self, frame):
        cv2.putText(frame, "Hitung: "+str(self.__left_vehicle), (100, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)
    
    # def putRightVehicleCounter(self, frame):
    #     cv2.putText(frame, "Kanan: "+str(self.__right_vehicle), (900, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0),5)

class Classification():
    def __init__(self, prototxt_path="MobileNetSSD_deploy.prototxt", model_path="MobileNetSSD_deploy.caffemodel"):
        self.__prototxt_path = prototxt_path
        self.__model_path = model_path
        self.__CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]
        print(self.__prototxt_path)
        self.__net = cv2.dnn.readNetFromCaffe(self.__prototxt_path, self.__model_path)

    def detect(self, frame):
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, size=(300, 300), ddepth=cv2.CV_8U)
        self.__net.setInput(blob, scalefactor=1.0/127.5, mean=[127.5, 127.5, 127.5])
        detection = self.__net.forward()

        for i in np.arange(0, detection.shape[2]):
             confidence = detection[0, 0, i, 2]
             if confidence > 0.1:
                idx = int(detection[0, 0, i, 1])
                self.__label = self.__CLASSES[idx]
                break
    
    def drawClassification(self, frame, x, y):
        cv2.putText(frame, self.__label,(x,y-50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)

class Speed():
    def __init__(self):
        self.__PTime = 0

    def setFps(self):
        self.__CTime = time.time()
        self.__fps = 1/(self.__CTime - self.__PTime)
        self.__PTime = self.__CTime
    
    def getFps(self):
        return self.__fps
    
    def setEstimate(self, x, y):
        d_pixels = math.sqrt(x + y)
        ppm = 8.8
        d_meters = int(d_pixels*ppm)
        self.__speed = d_meters/self.__fps*3.6
        self.__speed_in_km = np.average(self.__speed)

    def drawText(self, frame, x, y):
        cv2.putText(frame,str(int(self.__speed_in_km))+"Km/h",(x,y-15),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)

cap = cv2.VideoCapture("cars.mp4")
subtraction = cv2.createBackgroundSubtractorMOG2()

counter = Counter()
classification = Classification()
speed = Speed()

while True:
    ret , frame = cap.read()

    detection = Detection(frame, subtraction)
    detection.subtract()
    contour = detection.getContour()

    counter.drawLine(frame, 25, 1250, (0,255,0))
    #counter.drawLine(frame, 725, 1250, (0,0,255))
    for c in contour:
        detection.validateOutline(c)
        validate = detection.getValidateOutline()
        if validate == False:
            continue

        detection.drawRectangle()
        detection.drawDotCenter()
        detect = detection.getDetection()
        x, y, w, h = detection.getBoundingRect()

        #Speed
        speed.setFps()
        fps = speed.getFps()
        speed.setEstimate(x, y)
        speed.drawText(frame, x, y)

        #Classification
        classification.detect(frame)
        classification.drawClassification(frame, x, y)

        #Counter
        for(x,y) in detect:
            if counter.detecLeft(25, 1250, x, y):
                detection.removeDetect((x,y))
            elif counter.detecRight(725, 1250, x, y):
                detection.removeDetect((x,y))

    #Counter
    counter.putLeftVehicleCounter(frame)
    # counter.putRightVehicleCounter(frame)

    cv2.imshow("video" , frame)
    
    if cv2.waitKey(1) == ord('q'):
        break
    elif cv2.getWindowProperty('video',cv2.WND_PROP_VISIBLE) < 1:        
        break

cap.release()
cv2.destroyAllWindows()