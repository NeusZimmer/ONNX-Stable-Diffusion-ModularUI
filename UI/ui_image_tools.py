import gradio as gr
import gc,os
from Scripts import deepdanbooru_onnx as DeepDanbooru_Onnx
from Scripts import image_slicer
from Scripts import superresolution as SPR


global debug
global danbooru
global image_in
danbooru = None
debug = False


def show_danbooru_area():
    global image_in
    with gr.Row():
        with gr.Column(variant="compact"):
            image_in = gr.Image(label="input image", type="pil", elem_id="image_init")
    with gr.Row():
        apply_btn = gr.Button("Analyze image with Deep DanBooru", variant="primary")
        mem_btn = gr.Button("Unload from memory")

    with gr.Row():
        results = gr.Textbox(value="", lines=8, label="Results")
    mem_btn.click(fn=unload_DanBooru, inputs=results , outputs=results)
    apply_btn.click(fn=analyze_DanBooru, inputs=image_in , outputs=results)

def analyze_DanBooru(image):
    global danbooru
    if danbooru == None:
        danbooru = DeepDanbooru_Onnx.DeepDanbooru()
    results=danbooru(image)
    results2=str(results.keys())
    results2=results2.replace("'","")
    results2=results2.replace("dict_keys([","")
    results2=results2.replace("])","")
    #return list(results)
    return results2

def unload_DanBooru(results):
    global danbooru
    danbooru= None
    gc.collect
    return results+"\nUnloaded from memory of provider"

def show_image_resolution_area():
    with gr.Row():
        resolution_btn = gr.Button("Resize image with ONNX", variant="primary")
    with gr.Row():
        test="prueba"
        gr.Markdown("Choose number of divisions to be applied (higher means higher resolution)"+"\nCurrent Image Size"+test)
        #img_parts = gr.Slider(2, 50, value=4, label="Divisions", step=1)
        checkbox1 = gr.Checkbox(label="3 Steps", info="Try to reduce the slicing/join lines")
        img_rows = gr.Slider(2, 30, value=4, label="Row Divisions", step=1)
        img_cols = gr.Slider(2, 30, value=4, label="Column Divisions",step=1)
    with gr.Row():
        image_out = gr.Image(label="Output image", type="pil", elem_id="image_out")
    with gr.Row():
        Mem_btn = gr.Button("Clean SuperResolution Memory")
        delete_btn = gr.Button("Delete Temp dir")


    resolution_btn.click(fn=Resize_Image, inputs=[image_in,img_rows,img_cols,checkbox1], outputs=image_out)
    Mem_btn.click(fn=Clean_SPR_Mem, inputs=None, outputs=image_out)
    delete_btn.click(fn=DeleteTemp, inputs=None , outputs=None)

def ResizeAndJoin(row,col):
    for files in sorted(os.listdir("./Temp")):
        if "TemporalImage" in files:
            SPR.superresolution_process("./Temp/"+files)
    tiles=image_slicer.open_images_in("./Temp")
    img=image_slicer.join(tiles,row,col)
    return img

def Clean_SPR_Mem():
    SPR.clean_superresolution_memory()
    gc.collect()

def DeleteTemp():
    image_slicer.delete_temp_dir()

def adjustate_column_marks(img,image,img_cols,img_rows,pixels):
    halfof_tile_width_size=(image.size[0]/img_cols)/2
    left=halfof_tile_width_size
    right=image.size[0]-halfof_tile_width_size
    img_columnmarks_correction=image.crop((left,0,right,image.size[1]))
    image_slicer.slice(img_columnmarks_correction,row=img_rows,col=img_cols-1)
    img_columnmarks_correction= None
    for files in sorted(os.listdir("./Temp")):
        if "TemporalImage" in files:
            SPR.superresolution_process("./Temp/"+files)
            image_slicer.create_slice_of_substitute_for_column_joint("./Temp/"+files,pixels)
    tiles=image_slicer.open_images_in("./Temp")
    img=image_slicer.substitute_image_joint_vertical_marks(img,tiles,pixels,img_cols)
    return img

    

def adjustate_row_marks(img,image,img_cols,img_rows,pixels):
    halfof_tile_height_size=(image.size[1]/img_rows)/2
    left = 0
    right = image.size[0]
    top = halfof_tile_height_size
    down = image.size[1]-halfof_tile_height_size
    img_rowmarks_correction=image.crop((left,top,right,down))
    image_slicer.slice(img_rowmarks_correction,row=img_rows-1,col=img_cols)

    for files in sorted(os.listdir("./Temp")):
        if "TemporalImage" in files:
            SPR.superresolution_process("./Temp/"+files)
            image_slicer.create_slice_of_substitute_for_row_joint("./Temp/"+files,pixels)

    tiles=image_slicer.open_images_in("./Temp")
    img=image_slicer.substitute_image_joint_horizontal_marks(img,tiles,pixels,img_rows)
    return img

def Resize_Image(image,img_rows,img_cols,checkbox1):
    pixels=8  #Manual Adjustment
    #Adjust img size to col&rows multiples, and make sure both tile sizes are divisible by 2
    image=image_slicer.adjust_image_size(image,img_rows,img_cols)

    #tiles=image_slicer.slice(image,row=img_rows,col=img_cols)
    #tiles=None  #Modify previous to save memory if not used?
    image_slicer.slice(image,row=img_rows,col=img_cols)
    img=ResizeAndJoin(row=img_rows,col=img_cols)

    DeleteTemp()

    Mark_Adjustement = checkbox1
    if Mark_Adjustement:
        image=adjustate_column_marks(img,image,img_cols,img_rows,pixels)
        DeleteTemp()
        adjustate_row_marks(img,image,img_cols,img_rows,pixels)
        DeleteTemp()
        img=blur_Image(img)
        img=Sharpen_Image(img) #Create option for them, they delay too much the creation

    return img


def blur_Image(img):
    import cv2 as cv
    import numpy as np
    img = np.array(img)
    img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)
    blur = cv.GaussianBlur(img,(5,5),0)
    #blur = cv.medianBlur(img,5)
    #blur = cv.bilateralFilter(img,9,75,75)
    #blur=cv.blur(img,(5,5))
    return blur

def Sharpen_Image(img):
    import numpy as np
    import cv2 as cv
    img = np.array(img) 
    if is_grayscale(img):
        height, width = img.shape
    else:
        img = cv.cvtColor(img, cv.CV_8U)
        height, width, n_channels = img.shape

    result = np.zeros(img.shape, img.dtype)
    for j in range(1, height - 1):
        for i in range(1, width - 1):
            if is_grayscale(img):
                sum_value = 5 * img[j, i] - img[j + 1, i] - img[j - 1, i] \
                            - img[j, i + 1] - img[j, i - 1]
                result[j, i] = saturated(sum_value)
            else:
                for k in range(0, n_channels):
                    sum_value = 5 * img[j, i, k] - img[j + 1, i, k]  \
                                - img[j - 1, i, k] - img[j, i + 1, k]\
                                - img[j, i - 1, k]
                    result[j, i, k] = saturated(sum_value)
    
    return result


def is_grayscale(my_image):
    return len(my_image.shape) < 3

def saturated(sum_value):
    if sum_value > 255:
        sum_value = 255
    if sum_value < 0:
        sum_value = 0
    return sum_value

