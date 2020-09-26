#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.spatial import distance as dist
import cv2
import numpy as np
from fastai import *
from fastai.vision import *
import pandas as pd
import argparse
import imutils
from imutils.video import FileVideoStream
from imutils import face_utils
import dlib
import torch 

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video-file", required=True, help="video file in current directory")
ap.add_argument("--save", dest="save", action = "store_true")
ap.add_argument("--savedata", dest="savedata", action = "store_true")

ap.set_defaults(savedata = False)
ap.set_defaults(save = False)
args = vars(ap.parse_args())

path = "/Users/adyasingh/Desktop/body-language-detector/EmotionDetector/"
vidcap = FileVideoStream(args["video_file"]).start()
framecount = 0
learn = load_learner(path, 'model.pkl')
data = []
atten = []

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 40
prediction =''
COUNTER = 0
ALARM_ON = False
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

if args["save"]:
    out = cv2.VideoWriter(path + "output.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 10, (450,300))

while vidcap.more():
    frame = vidcap.read()
    if frame is None:
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coord = face_cascade.detectMultiScale(gray, 1.1, 20, minSize=(30, 30))
    for coords in face_coord:
        X, Y, w, h = coords
        H, W, _ = frame.shape
        X_1, X_2 = (max(0, X - int(w * 0.3)), min(X + int(1.3 * w), W))
        Y_1, Y_2 = (max(0, Y - int(0.3 * h)), min(Y + int(1.3 * h), H))
        img_cp = gray[Y_1:Y_2, X_1:X_2].copy()
        rect = dlib.rectangle(X, Y, X+w, Y+h)
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
       
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "Low Attention", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if framecount % 10 == 0:
                    atten.append([framecount,"Low"])
        else:
            COUNTER = 0
            ALARM_ON = False
            if framecount % 10 == 0:
                atten.append([framecount,"Normal"])
       
        if framecount % 10 == 0:
            prediction, idx, probability_ten = learn.predict(Image(pil2tensor(img_cp, np.float32).div_(225)))
            probability = probability_ten.tolist()
            confident = probability[0]
            enthusiastic  = probability[1]
            frustrated = probability[2]
            nervous = probability[3]
            neutral = probability[4]
            scared = probability[5]
            uncomfortable = probability[6]
            
            data.append([framecount, prediction, confident,enthusiastic,frustrated,nervous,neutral,scared,uncomfortable])

    
        cv2.putText(frame, str(prediction), (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 255, 255), 2)

    cv2.imshow("frame", frame)
    framecount += 1
    if args["save"]:
        out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if args["savedata"]:
    bodyLanguageFile = "/BL_"+ args["video_file"] +".csv"
    attentionFile = "/A_"+ args["video_file"] +".csv"
    df = pd.DataFrame(data, columns = ['Framecount', 'Prediction','Confident', 'Enthusiastic', 'Frustrated', 'Nervous', 'Neutral', 'Scared', 'Uncomfortable'])
    df.to_csv(path+bodyLanguageFile)
    df2 = pd.DataFrame(atten, columns = ['Framecount', 'Attention'])
    df2.to_csv(path+attentionFile)
    print("data saved!")
vidcap.stop()
if args["save"]:
    print("done saving")
    out.release()
cv2.destroyAllWindows()
