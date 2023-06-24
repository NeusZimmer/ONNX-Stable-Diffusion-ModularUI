#Class code, initial model and tags obtained from 


import onnxruntime as ort
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import requests
import hashlib
from typing import List, Union
import shutil
from pathlib import Path
import hashlib
from Engine.General_parameters import Engine_Configuration
import cv2

def download_model():
    #return "./Scripts/correction-codeformer.onnx"
    #return "./Scripts/correction-gfpgan-v1-3.onnx"
    #Download model from : https://huggingface.co/Neus/GFPGANv1.4/
    return "./Scripts/GFPGANv1.4.onnx"

class CodeFormer:
    def __init__(self, mode: str = "auto"):
        '''
        Initialize the  class.
         '''
        exec_provider=Engine_Configuration().DeepDanBooru_provider
        providers = {
            "cpu":"CPUExecutionProvider",
            #"gpu":"CUDAExecutionProvider",
            "gpu":'DmlExecutionProvider',
            #"tensorrt": "TensorrtExecutionProvider",
            #"auto":"CUDAExecutionProvider" if "CUDAExecutionProvider" in ort.get_available_providers() else "CPUExecutionProvider",
            "auto":exec_provider,
            #"auto":'DmlExecutionProvider',
        }

        if mode not in providers:
            raise ValueError("Mode not supported. Please choose from: cpu, gpu, tensorrt")
        if providers[mode] not in ort.get_available_providers():
            raise ValueError(f"Your device is not supported {mode}. Please choose from: cpu")

        model_path = download_model()

        #self.session = ort.InferenceSession(model_path, providers=[providers[mode]])
        self.session = ort.InferenceSession(model_path, providers=['DmlExecutionProvider', 'CPUExecutionProvider'])

        self.input_name = self.session.get_inputs()[0].name
        print(f"Shape:{self.session.get_inputs()[0].shape}")
        print(f"Long salida:{len(self.session.get_outputs())}")
        print("Sessiones inicio")
        print(self.session.get_inputs()[0])
        print(self.session.get_inputs())
        self.output_name = [output.name for output in self.session.get_outputs()]
        self.mode = mode
        self.cache = {}

    def __str__(self) -> str:
        return f"Codeformer(mode={self.mode}, inputs:{self.input_name }, outputs:{self.output_name})"

    def __repr__(self) -> str:
        return self.__str__()

    def __call__(self, image):
        return self.process_image(image)

    def preprocess(self,img):
        #import cv2
        import numpy as np
        newsize = (512, 512)
        img = img.resize(newsize)
        img = np.asarray(img)#.astype('float32')
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = img / 255.0

        img[:,:,0] = (img[:,:,0]-0.5)/0.5
        img[:,:,1] = (img[:,:,1]-0.5)/0.5
        img[:,:,2] = (img[:,:,2]-0.5)/0.5

        img = np.float32(img[np.newaxis,:,:,:])
        img = img.transpose(0, 3, 1, 2)
        return img


    def process_image(self,image):
        import cv2
        image = self.preprocess(image)
        #dict={self.input_name:torch_img.cpu().numpy()}
        dict={self.input_name:image}
        output=self.session.run(self.output_name, dict)[0]

        output = output[0]
        output = output.clip(0,1)
        output = output.transpose(1, 2, 0)

        #output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        output = (output + 1) / 2
        output = (output * 255.0).round()
        img = output.astype(np.uint8)
        #print(output)
        return img

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    from PIL import Image
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]
    return pil_images





class GFPGANFaceAugment:
    def __init__(self, model_path, use_gpu = False):
        self.ort_session = ort.InferenceSession(model_path,providers=('DmlExecutionProvider', 'CPUExecutionProvider'))
        self.net_input_name = self.ort_session.get_inputs()[0].name
        _,self.net_input_channels,self.net_input_height,self.net_input_width = self.ort_session.get_inputs()[0].shape
        self.net_output_count = len(self.ort_session.get_outputs())
        self.face_size = 512
        self.face_template = np.array([[192, 240], [319, 240], [257, 371]]) * (self.face_size / 512.0)
        self.upscale_factor = 2
        self.affine = False
        self.affine_matrix = None

    def pre_process(self, img):
        img = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)))
        img = cv2.resize(img, (self.face_size, self.face_size))
        img = img / 255.0
        img = img.astype('float32')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img[:,:,0] = (img[:,:,0]-0.5)/0.5
        img[:,:,1] = (img[:,:,1]-0.5)/0.5
        img[:,:,2] = (img[:,:,2]-0.5)/0.5
        img = np.float32(img[np.newaxis,:,:,:])
        img = img.transpose(0, 3, 1, 2)
        return img

    def post_process(self, output, height, width):
        output = output.clip(-1,1)
        output = (output + 1) / 2
        output = output.transpose(1, 2, 0)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        output = (output * 255.0).round()
        if self.affine:
            inverse_affine = cv2.invertAffineTransform(self.affine_matrix)
            inverse_affine *= self.upscale_factor
            if self.upscale_factor > 1:
                extra_offset = 0.5 * self.upscale_factor
            else:
                extra_offset = 0
            inverse_affine[:, 2] += extra_offset
            inv_restored = cv2.warpAffine(output, inverse_affine, (width, height))
            mask = np.ones((self.face_size, self.face_size), dtype=np.float32)
            inv_mask = cv2.warpAffine(mask, inverse_affine, (width, height))
            inv_mask_erosion = cv2.erode(
                inv_mask, np.ones((int(2 * self.upscale_factor), int(2 * self.upscale_factor)), np.uint8))
            pasted_face = inv_mask_erosion[:, :, None] * inv_restored
            total_face_area = np.sum(inv_mask_erosion)
            # compute the fusion edge based on the area of face
            w_edge = int(total_face_area**0.5) // 20
            erosion_radius = w_edge * 2
            inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
            blur_size = w_edge * 2
            inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)
            inv_soft_mask = inv_soft_mask[:, :, None]
            output = pasted_face
        else:
            inv_soft_mask = np.ones((height, width, 1), dtype=np.float32)
            output = cv2.resize(output, (width, height))
        return output, inv_soft_mask

    def forward(self, img):
        import PIL
        img = np.asarray(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        height, width = img.shape[0], img.shape[1]
        img = self.pre_process(img)
        ort_inputs = {self.ort_session.get_inputs()[0].name: img}
        ort_outs = self.ort_session.run(None, ort_inputs)
        output = ort_outs[0][0]
        output, inv_soft_mask = self.post_process(output, height, width)
        output = output.astype(np.uint8)

        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        output = PIL.Image.fromarray(output)
        return output#, inv_soft_mask


