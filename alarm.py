
from __future__ import division
import dlib
from imutils import face_utils
import cv2
import numpy as pp
from scipy.spatial import distance as dist
import threading
import pygame
import time
from datetime import datetime

alarmtime = "12:02"


def sound_on():
    pygame.mixer.init()
    pygame.mixer.music.load("z.ogg")
    pygame.mixer.music.play()

def sound_off():
    pygame.mixer.stop()

def resize(img, width=None, height=None, interpolation=cv2.INTER_AREA):
    global ratio
    w, h = img.shape
    if width is None and height is None:
        return img
    elif width is None:
        ratio = height / h
        width = int(w * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized
    else:
        ratio = width / w
        height = int(h * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized

def shape_pp(shape, dtype="int"):
    coop = pp.zeros((68, 2), dtype=dtype)
    for i in range(36,48):
        coop[i] = (shape.part(i).x, shape.part(i).y)
    return coop
def eye_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
 
	
    C = dist.euclidean(eye[0], eye[3])
   

    ear = (A + B) / (2.0 * C)
 

    return ear
camera = cv2.VideoCapture(0)

predictor_path = 'shape_predictor_68_face_landmarks.dat_2'

detector = dlib.get_frontal_face_detector() #use dlib libary to detect face
predictor = dlib.shape_predictor(predictor_path)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] #detect left right eyes
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
total=0

while True:
    ret, frame = camera.read()
    lcltime = datetime.now().strftime('%H:%M')
    if ret == False:
        print('Failed to capture frame from camera. Check camera index in cv2.VideoCapture(0) \n')
        break

    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_resized = resize(frame_grey, width=120)


    dets = detector(frame_resized, 1)
    
    if len(dets) > 0:
        for k, d in enumerate(dets):
            shape = predictor(frame_resized, d)
            shape = shape_pp(shape)
            lEyes= shape[lStart:lEnd]
            rEyes= shape[rStart:rEnd]
            lEAR= eye_ratio(lEyes)
            rEAR = eye_ratio(rEyes)
            drownResult = (lEAR + rEAR) / 2.0
            leftEyeHull = cv2.convexHull(lEyes)
	       
            rightEyeHull = cv2.convexHull(rEyes)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            

            if drownResult>.30:
                print (drownResult)
                total=0
                alarm=False
                cv2.putText(frame, "Eyes Open ", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                total+=1
                if total>20:
                    if not alarm:
                        alarm=True
                        d=threading.Thread(target=sound_on)
                        d.setDaemon(True)
                        d.start()
                        cv2.putText(frame, "You are sleepy" ,(250, 30),cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 0), 4)
                        if drownResult>.30:
                            print (drownResult)
                            total=0
                            alarm=False
                            d=threading.Thread()
                            d.stop()
                cv2.putText(frame, "Sleepy eyes".format(total), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            for (x, y) in shape:
                cv2.circle(frame, (int(x/ratio), int(y/ratio)), 3, (255, 255, 255), -1)
    cv2.imshow("image", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        camera.release()
        break
