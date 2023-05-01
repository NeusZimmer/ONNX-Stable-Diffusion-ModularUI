from PIL import Image
#from resizeimage import resizeimage
import numpy as np
import onnxruntime

global ort_session
ort_session = None

def superrsolution_pre_process_img(img_path):
    orig_img = Image.open(img_path)
    ratio= orig_img.size[0]/orig_img.size[1]
    img =orig_img.resize([224, 224], Image.Resampling.LANCZOS)
    img_ycbcr = img.convert('YCbCr')
    img_y_0, img_cb, img_cr = img_ycbcr.split()
    img_ndarray = np.asarray(img_y_0)
    img_4 = np.expand_dims(np.expand_dims(img_ndarray, axis=0), axis=0)
    img_5 = img_4.astype(np.float32) / 255.0
    return img_5, img_cb, img_cr,ratio


def superrsolution_post_process_img(img_out_y, img_cb, img_cr,ratio=1):
    img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')
    # get the output image follow post-processing step from PyTorch implementation
    final_img = Image.merge(
        "YCbCr", [
            img_out_y,
            img_cb.resize(img_out_y.size, Image.BICUBIC),
            img_cr.resize(img_out_y.size, Image.BICUBIC),
        ]).convert("RGB")
    final_img=final_img.resize([int(final_img.size[0]*ratio), final_img.size[1]])
    return final_img

def superresolution_process(img_path):
    img_5, img_cb, img_cr, ratio = superrsolution_pre_process_img(img_path)
    global ort_session
    if ort_session == None:
        ort_session = onnxruntime.InferenceSession("./Scripts/super-resolution/super-resolution-10.onnx",providers=['DmlExecutionProvider'])
    ort_inputs = {ort_session.get_inputs()[0].name: img_5} 
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = ort_outs[0]
    img_out_y = superrsolution_post_process_img(img_out_y, img_cb, img_cr,ratio)
    img_out_y.save(img_path)

def clean_superresolution_memory():
    global ort_session
    ort_session == None

