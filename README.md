# Deprecated, for new version go to: https://github.com/NeusZimmer/ONNX-ModularUI-StableDiffusion

# ONNX-ModularUI
**New version available**
**This version only works until diffusers version 14.0, does not work with new versions of diffusers, may need to modify the requirements.txt to make it run**



Hello, I'm a **AMD user with a low-profile 4gb GPU**.... i gone crazy trying to find a solution to make stable diffusion work as fast as possible. I only found a viable option: ONNX on Windows, found it and  started to try and test models while trying to optimize the in-memory model comsumption and performance/balance on the available UIs.

This version might work for CUDA, TensorRT, DirectML engines of ONNX, on windows and linux ( for linux you will need to modify the requirements.txt file and delete "onnxruntime-directml" line to make it work, NOTE: Maybe AMD cards not supported in linux implementation?)

I've just decided to apply some of them into this UI to allow a granular approach on how the models are loaded and how they consum memory & disk, while adding some options that other versions already had. Current version avoids the necessity of storing repeated Vae's, Text Encoders and ControlNet models... saving at least 0'5Gb per model

The Stable Diffusion pipelines classes are encapsulated in new ones to allow them an easier management while adding other options around.

**Current version:**

**-New Hi-Res pipeline & approach, plus latents experimentals ---A must try!**

**-Main basic pipelines: Txt2Img, Img2Img, Pix2Pix, Inpaint, ControlNet**

**-Additional tools: 2 upscalers, deepdanbooru tagging, face detection, wildcards support,styles**
	
**-Experimental feature: latents experimentals & image composition: creating one image using a sumatory of previous outputs, (one or many), working as something inbetween outpainting, img2img and controlnet.(works so good with the hi-res pipeline**

**Working features: You decide where to run&load each feature (model, vae, textenc...) based on your hardware , CUDA, DML, CPU... whithout the need of reseting the UI, only loading a new model or reloading current model. Avoid reusing of repeated models for VAE's, TextEncoder & ControlNet...**
**And: wilcards (for one or multiple iterations, also you could inclue wildcards inside other wildcards...), styles (examples available, editing in config tab.** 

Next version update: clip-skip

## Set up

To make this works follow the (shown below) install steps for ONNX Stabble diffusion.

```
git clone https://github.com/NeusZimmer/ONNX-Stable-Diffusion-ModularUI.git
```

Then :
```
pip install virtualenv
python -m venv sd_env
sd_env\scripts\activate
python -m pip install --upgrade pip
pip install torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu --pre
pip install -r requirements.txt
```

PD: to install in linux, use **pip install -r requirements-linux.txt** instead "pip install -r requirements.txt".

Activate python virtual environment and run the UI
```
./sd_env/scripts/activate.bat
cd ONNX-Stable-Diffusion-ModularUI
run.bat
or 
py -O ONNX-StableDiffusion.py
```

**Model Download**
I've uploaded an initial set of files for testing the UI into https://civitai.com/models/125580

**For Model conversion, please, use this repository:**
https://github.com/Amblyopius/Stable-Diffusion-ONNX-FP16

PD: do not need to create a new python environment, as this UI uses the same environment and there's no need to duplicate.


## Configuration
At first run, you may need to configure the path for some options: output & models, 
![UIOptions](https://github.com/NeusZimmer/ONNX-Stable-Diffusion-ModularUI/assets/94193584/a160aacd-39ca-4ab4-b75b-3e7f4d0ff82c)

and have a look into the Engine Configuration to see available option for running the different modules & pipelines.
![EngineOptions](https://github.com/NeusZimmer/ONNX-Stable-Diffusion-ModularUI/assets/94193584/08d40866-d472-40b2-a001-5cf7a9d8513b)


Currently, you may also select to use a specific VAE model to use for the inferences, saving some space on disk, as many models use the same versions, it will search for the first option, if not found, it will go for 2nd and then for 3rd...
This also applies for ControlNet models, you could use the same one for all instances of your model, saving a lot of space on your disk.Note: you will still need the adapted Unet in the ControlNet directory.
![VAEOptions](https://github.com/NeusZimmer/ONNX-Stable-Diffusion-ModularUI/assets/94193584/6232335f-9442-482b-ba0d-eca79c2bc09a)



You may check the info on the cmd window about what and where a model have been loaded:

![CMD-Example](https://github.com/NeusZimmer/ONNX-Stable-Diffusion-ModularUI/assets/94193584/4151131a-5fe3-43a8-bb52-9360ed471127)


As you may see in the directory tree of one of my models, there's no need for VAE directories and also for the generic ControlNet Models:
![ExampleControlNetModel](https://github.com/NeusZimmer/ONNX-Stable-Diffusion-ModularUI/assets/94193584/431144a8-77e9-41b3-8525-d8c3388e9f22)


From previous readme: to be edited:

It provides support for: txt2img, img2img, Inpaint, instruct pix2pix.(control net in test phase), and NEW: reingest of a previous generation into a new one, it works closely to ControlNet without models ...

Allows modification of the pipeline providers without re-running the UI., also, you may want to run some pipeline in one graphic card, another card for VAE and CPU to the rest... this UI allows such granularity for :main model, schedulers, VAE, text encoder...

Allows the use of a different VAE for a model (many models got the same VAE , then, why keep storing them on disk?

Add a clean memory option: changing resolution for the inferences keep garbage in memory, and it ends making an impact on the time needed for the inferences.

Also: wildcards, live prompts while running multiple iterations, a working deepdanbooru interrogator and resolution increase option. (one of my first tests with a ONNX model, a MS Model ([super-resolution-10.onnx]) to increase file resolution up to crazy sizes by slicing & joining (working , but not for professional uses) (todo: implement stable diffusion 4x upscale...)

For DeepDanbooru model, download it from: https://huggingface.co/Neus/Onnx_DeepDanbooru/tree/main

PD: first python code I did... sure it could be improved, but working fine and easy understand its workflows and to modify in future releases.

1st-Mayor updated, i was looking for an option to re-use previous generated latents of images, and i did one approach, basing on their generated latents, available under txt2img tab, accordion: Latent Experimentals. For one or sumatories of latens... look and the info txt file to make it work.

Works fairly well, but only with DDIMM (ETA=1) and Euler Ancestral, while other schedulers are hard to find the right conbination (or impossible), with DDIM & Euler A you may get pretty good definition and images based on a previous generation, from same model or from a different model.

1st. find a prompt, model and a generation that you want to save and use
![SaveLatents1](https://github.com/NeusZimmer/ONNX-ModularUI/assets/94193584/5778f303-d9ef-4dcb-8cd6-74a7c8998359)

It will save a .npy file in the latents directory, one file each two steps, and the final one.
![latents2](https://github.com/NeusZimmer/ONNX-ModularUI/assets/94193584/5fef7606-ba1e-4e43-ab19-04f0aeb3ee8e)

If you want to check the result image of each .npy file, you may click on "Convert npy to img" and it will save .png alongside the numpy files. (I always use the last one, but feel free to try with previous...)
![Optional-CheckImages](https://github.com/NeusZimmer/ONNX-ModularUI/assets/94193584/76e610cd-64b7-4121-a53d-56ece339e6e3)
![latents+imgs](https://github.com/NeusZimmer/ONNX-ModularUI/assets/94193584/8cb7ffff-15be-4aa5-b6b9-93b9834eae1f)


Write down the name of the numpy to use for the next generated image and a different promt, (having in mind the constrains of the previously generated image)
For only one latent use: "index:name.npy" formula: index (ie: "1:file.npy"  formula:"1")
PD: Make sure the size (width and height) of the previous image is according to the new generation
![ExampleOfSums](https://github.com/NeusZimmer/ONNX-Stable-Diffusion-ModularUI/assets/94193584/e572d604-e7f0-4343-b4dc-f1322169bb47)

Here is the tricky part, Select Euler A, a multiplier ( best results range between 0.35 to 0.6 ) and a Strengh, strengh is the total steps you will be applying the numpy file, if you go fot 36 steps a good number could range from 28 to 32 (from 70% to 90%).
Steps: you could have a good approach of the final image from 14-16 steps, and good results around 30 to 40 steps.
![re-generate](https://github.com/NeusZimmer/ONNX-ModularUI/assets/94193584/03afe051-ec35-438a-abcd-2e401f1bd4e6)
Guid: depends on the model,for some I found the need to increase it a little bit, 10 to 18 could be a good starting point (12-14 the usual), but others works as usual...


