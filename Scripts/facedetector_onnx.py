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
                "gpu":"DMLExecutionProvider",
                "cpu":"CPUExecutionProvider",
                "gpu":"CUDAExecutionProvider",
                "auto":exec_provider,
            }
        face_detector_onnx = "./Scripts/facedetector/version-RFB-320.onnx"
        face_detector_onnx = "./Scripts/facedetector/version-RFB-640.onnx"
        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level=3
        self.face_detector = ort.InferenceSession(face_detector_onnx,providers=providers,sess_options=sess_options)

    def face_restore(self,face):
        print("Face restoring do not implemented yet")
        return face

    def paste_face(self,img_with_boxes,box,restored_face):
        from PIL import Image
        #print("Entrando en pegar cara")
        Image.Image.paste(img_with_boxes, restored_face, (box[0],box[1]))
        return img_with_boxes 

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
        height, width, _ = image.shape
        for i in range(boxes.shape[0]):
            box = self.scale(boxes[i, :],width,height)
            #box =boxes[i, :]
            face_img=self.cropImage(orig_image,box)
            faces.append(face_img)
            cv2.rectangle(orig_image2, (box[0], box[1]), (box[2], box[3]), color, 1)
            #cv2.imwrite(f"./faces/face{i}.png",face_img)
            boxes[i]=box

        from PIL import Image
        orig_image2 = cv2.cvtColor(orig_image2, cv2.COLOR_BGR2RGB)
        orig_image2 = Image.fromarray(orig_image2)
        faces2=[]
        for face in faces:
            face2 = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            faces2.append(Image.fromarray(face2))

        return orig_image2,faces2,boxes

    def __str__(self) -> str:
        return f"FaceDetector" #(mode={self.mode}, threshold={self.threshold}, pin_memory={self.pin_memory}, batch_size={self.batch_size})"

    def __repr__(self) -> str:
        return self.__str__()

    def __call__(self, image):
        return self.run_session(image)

    # scale current rectangle to box
    def scale(self,box,width0,height0):
        normal=((width0+height0)/4)/10
        box[0]= 0 if (box[0]-normal)<0 else (box[0]-normal)
        box[1]= 0 if (box[1]-normal)<0 else (box[1]-normal)
        box[2]= width0-1 if (box[2]+normal)>width0-1 else (box[2]+normal)
        box[3]= height0-1 if (box[3]+normal)>height0-1 else (box[3]+normal)


        width = box[2] - box[0]
        height = box[3] - box[1]
        #print(f"width:{width}")
        #print(f"height:{height}")
        maximum = max(width, height)
        if (box[2]+maximum)>width0-1:maximum = width
        if (box[3]+maximum)>height0-1:maximum = height
        #dx = int((maximum - width)/2)
        #dy = int((maximum - height)/2)
        #print(box)
        bboxes = [box[0], box[1], box[0]+maximum , box[1]+maximum]
        #bboxes = [box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy]
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

    def select_box(self,orig_image,point_x, point_y,side_size=128):
        import numpy as np
        import cv2

        width, height = orig_image.size
        orig_image = orig_image.convert('RGB') 
        orig_image = np.array(orig_image) 
        # Convert RGB to BGR 
        orig_image2 = orig_image[:, :, ::-1].copy() 
        orig_image3 = orig_image2.copy() 
        color = (255, 128, 0) #rectangle color

        point_x= point_x if (point_x+side_size<width) else (width-(side_size+1))
        point_y= point_y if (point_y+side_size<height) else (height-(side_size+1))

        box=np.array([point_x, point_y,point_x+side_size, point_y+side_size])
        area= self.cropImage(orig_image3, box)
        cv2.rectangle(orig_image2, (box[0], box[1]), (box[2], box[3]), color, 1)
        print(f"Esto es boxes:{type(box)}")

        from PIL import Image
        orig_image2 = cv2.cvtColor(orig_image2, cv2.COLOR_BGR2RGB)
        orig_image2 = Image.fromarray(orig_image2)

        area = cv2.cvtColor(area, cv2.COLOR_BGR2RGB)
        area = Image.fromarray(area)

        return orig_image2,box,area


    def unload(self):
        self.face_detector=None


