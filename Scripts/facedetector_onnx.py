#Class code, initial model and tags obtained from https://huggingface.co/chinoll
#new model version converted on: 


"""import onnxruntime as ort
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import requests
import hashlib
from typing import List, Union
import shutil
from pathlib import Path
import hashlib"""

import cv2
import onnxruntime as ort
#import argparse   #no utilizaremos args, vendra una imagen directamente.
import numpy as np
from Scripts.facedetector.box_utils import predict
from Engine.General_parameters import Engine_Configuration



"""def download_model():
    return "./Scripts/deepdanbooru_onnx_data/deepdanbooru.onnx", "./Scripts/deepdanbooru_onnx_data/tags.txt"
"""


class FaceDetector:
    face_detector = None
    def __init__(self):
        if self.face_detector == None:
            exec_provider=Engine_Configuration().DeepDanBooru_provider
            providers = {
                "cpu":"CPUExecutionProvider",
                "gpu":"CUDAExecutionProvider",
                "auto":exec_provider,
            }
        face_detector_onnx = "./Scripts/facedetector/version-RFB-320.onnx"
        face_detector_onnx = "./Scripts/facedetector/version-RFB-640.onnx"
        self.face_detector = ort.InferenceSession(face_detector_onnx,providers=providers)


    def run_session(self,image):
        #convert to cv2
        image = image.convert('RGB') 
        image = np.array(image) 
        # Convert RGB to BGR 
        orig_image = image[:, :, ::-1].copy() 
        orig_image2 = orig_image.copy()
        color = (255, 128, 0) #rectangle color
        boxes, labels, probs = self.faceDetector(orig_image2)
        faces=[]
        for i in range(boxes.shape[0]):
            box = self.scale(boxes[i, :])
            #box =boxes[i, :]
            print(box)
            box=(box[0]-20,box[1]-20,box[2]+20,box[3]+20)
            face_img=self.cropImage(orig_image,box)
            faces.append(face_img)
            cv2.imwrite(f"./faces/face{i}.png",face_img)
        for i in range(boxes.shape[0]):
            box = self.scale(boxes[i, :])
            #box =boxes[i, :]
            cv2.rectangle(orig_image2, (box[0], box[1]), (box[2], box[3]), color, 2)
            #cv2.imshow('', orig_image)
            #cv2.imwrite("test.png",orig_image)

        from PIL import Image
        orig_image2 = cv2.cvtColor(orig_image2, cv2.COLOR_BGR2RGB)
        orig_image2 = Image.fromarray(orig_image2)
        faces2=[]
        for face in faces:
            face2 = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            faces2.append(Image.fromarray(face2))

        return orig_image2,faces2

    def __str__(self) -> str:
        return f"FaceDetector" #(mode={self.mode}, threshold={self.threshold}, pin_memory={self.pin_memory}, batch_size={self.batch_size})"

    def __repr__(self) -> str:
        return self.__str__()

    def __call__(self, image):
        return self.run_session(image)

    # scale current rectangle to box
    def scale(self,box):
        width = box[2] - box[0]
        height = box[3] - box[1]
        maximum = max(width, height)
        dx = int((maximum - width)/2)
        dy = int((maximum - height)/2)

        bboxes = [box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy]
        return bboxes

    # crop image
    def cropImage(self,image, box):
        num = image[box[1]:box[3], box[0]:box[2]]
        return num

    # face detection method
    def faceDetector(self,orig_image, threshold = 0.7):
        image = orig_image
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        #image = cv2.resize(image, (320, 240))
        image = cv2.resize(image, (640, 480))
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)

        input_name = self.face_detector.get_inputs()[0].name
        confidences, boxes = self.face_detector.run(None, {input_name: image})
        boxes, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)
        return boxes, labels, probs

    def unload(self):
        self.face_detector=None