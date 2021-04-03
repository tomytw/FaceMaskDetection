# import the necessary packages

# tensorflow pakai yang versi 2.3 biar bisa import model yang saya buat di colab

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
from os import listdir
from os.path import isfile, join, basename

import sys

def detect_and_predict_mask(frame, faceNet, maskNet):

    # reset faceNet karena sebelumnya sudah diapakai untuk blob yang width dan height berbeda
    if(not useVideo):
        faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]

    # blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
        # (104.0, 177.0, 123.0))
    blob = cv2.dnn.blobFromImage(frame, 1.0, (w, h),
        (104.0, 177.0, 123.0),crop=False)

    if(useVideo):
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
            (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]
        

        # untuk gambaran minConfidence nay dibuat 0.3 agar dapat mendeteksi kepala yang kecil digambar
        minConfidence = 0.5 if useVideo else 0.3

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > minConfidence:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

            
            check = box>600;
            # print(check)
            if True in check:
                continue
            (startX, startY, endX, endY) = box.astype("int")

            # membuat bounding box lebih besar agar masker lebih terlihat dengan sempurna
            boxWidth = endX - startX
            boxHeight = endY - startY

            if(not useVideo):
                startX , startY = np.subtract([startX,startY],int(boxWidth*0.2))
                endX, endY = np.add([endX,endY],int(boxHeight*0.2))
            else:
                startX , startY = np.subtract([startX,startY],int(boxWidth*0.1))
                endX, endY = np.add([endX,endY],int(boxHeight*0.1))
            

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))

            # # checking area face apakah sudah benar
            # cv2.imshow("Face", face)
            # key = cv2.waitKey(0)

            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
        # print(preds)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)

# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load model dari hasil training epoch ke 20
maskNet = load_model("save-state/model-020.h5")


# ubah menjadi True jika ingin menggunakan live feed video dari webcam
useVideo = False
showFalseBox = True

imageInputDir = "input/"

if (not useVideo):
    print("[INFO] starting detecting image...\n")

    inputFiles = [f for f in listdir(imageInputDir) if isfile(join(imageInputDir, f))]

    for f in inputFiles:

        imageFilePath = imageInputDir+f

        print("=====================================")
        print("DETEKSI IMAGE : ",imageFilePath)

        # imageFilePath = "dataset/mask-2.jpg"
        frame = cv2.imread(imageFilePath)
        frame = imutils.resize(frame,width=600)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding
        # locations
        jumlahMuka = len(preds)

        details = "Jumlah Muka: {} \n".format(jumlahMuka)

        wearingMask = 0

        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            if (not showFalseBox) and (withoutMask>mask) : continue

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # tambahkan wearingMask jika terdeteksi wajah pakai masker
            if mask > withoutMask : wearingMask += 1

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_DUPLEX, 0.4, [0,0,0], 2, cv2.LINE_AA)
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_DUPLEX, 0.4, color, 1, cv2.LINE_AA)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)


        # tampilkan jumlah yang pakai masker dan tidak pakai masker
        details += "Jumlah Orang yang pakai masker: {} \n".format(wearingMask)
        details += "Jumlah Orang yang tidak pakai masker: {} \n".format(jumlahMuka - wearingMask)

        print(details)

        # show the output frame
        cv2.imshow(imageFilePath, frame)

        # diganti dengan waitkey 0 soalnya cuman nampilin gambar aja
        print("Tekan Q untuk lanjut")
        print("=====================================")
        print("\n")
        key = 0
        while(key != ord("q")):
            key = cv2.waitKey(0)

        # simpan output frame ke folder output
        fileName = basename(imageFilePath)
        cv2.imwrite('output/'+fileName,frame)

        cv2.destroyAllWindows()

else:
    print("[INFO] starting video stream...")
    # loop over the frames from the video stream
    vs = VideoStream(src=0).start()
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        # frame = cv2.imread("dataset/mask-11.jpg")
        frame = imutils.resize(frame,width=600)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding
        # locations
        jumlahMuka = len(preds)
        print("Jumlah Muka", jumlahMuka)

        wearingMask = 0

        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # tambahkan wearingMask jika terdeteksi wajah pakai masker
            if mask > withoutMask : wearingMask += 1

            # display the label and bounding box rectangle on the output
            # frame

            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)


        # tampilkan jumlah yang pakai masker dan tidak pakai masker
        print("Jumlah Orang yang pakai masker: ",wearingMask)
        print("Jumlah Orang yang tidak pakai masker: ",jumlahMuka - wearingMask)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            vs.stream.release()
            vs.stop()
            break

# do a bit of cleanup
cv2.destroyAllWindows()

