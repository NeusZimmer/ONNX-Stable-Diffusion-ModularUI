# ONNX-ModularUI
An approach to make a more modular UI for ONNX Stable Diffusion, to ease future upgrades

To make it work
Follow install steps from :https://github.com/Amblyopius/Stable-Diffusion-ONNX-FP16/ and use it as your virtual environment for this code.

Clone/Copy this repository on a directory of your preference, activate your python environment, and run "py ONNX-StableDiffusion.py"

Current version provides support for: txt2img, img2img, Inpaint, instruct pix2pix.(control net in test phase), and NEW: reingest of a previous generation into a new one

Allows modification of the pipeline providers without re-running the UI., also, you may want to run some pipeline in one graphic card, another card for VAE and CPU to the rest... this UI allows such granularity for :main model, schedulers, VAE, text encoder...

Allows the use of a different VAE for a model (many models got the same VAE , then, why keep storing them on disk?

Add a clean memory option: changing resolution for the inferences keep garbage in memory, and it ends making an impact on the time needed for the inferences.

Also: wildcards, live prompts while running multiple iterations, a working deepdanbooru interrogator and resolution increase option. (one of my first tests with a ONNX model, a MS Model ([super-resolution-10.onnx]) to increase file resolution up to crazy sizes by slicing & joining (working , but not for professional uses)

For DeepDanbooru model, download it from: https://huggingface.co/Neus/Onnx_DeepDanbooru/tree/main

PD: first python code I did... sure it could be improved, but working fine and easy understand its workflows and to modify in future releases.

1st-Mayor updated, i was looking for an option to re-use previous generated images, and i did one approach, basing on their generated latents, available under txt2img tab, accordion: Latent Experimentals.

Works fairly well, but only with Euler Ancestral, while other schedulers are hard to find the right conbination (or impossible), with Euler A you may get pretty good definition and images based on a previous generation, from same model or from a different model.

1st. find a prompt, model and a generation that you want to save and use
![SaveLatents1](https://github.com/NeusZimmer/ONNX-ModularUI/assets/94193584/5778f303-d9ef-4dcb-8cd6-74a7c8998359)

It will save a .npy file in the latents directory, one file each two steps, and the final one.
![latents2](https://github.com/NeusZimmer/ONNX-ModularUI/assets/94193584/5fef7606-ba1e-4e43-ab19-04f0aeb3ee8e)

If you want to check the result image of each .npy file, you may click on "Convert npy to img" and it will save .png alongside the numpy files. (I always use the last one, but feel free to try with previous...)
![Optional-CheckImages](https://github.com/NeusZimmer/ONNX-ModularUI/assets/94193584/76e610cd-64b7-4121-a53d-56ece339e6e3)
![latents+imgs](https://github.com/NeusZimmer/ONNX-ModularUI/assets/94193584/8cb7ffff-15be-4aa5-b6b9-93b9834eae1f)


Write down the name of the numpy to use for the next generated image and a different promt, (having in mind the constrains of the previously generated image)
PD: Make sure the size (width and height) of the previous image is according to the new generation

Here is the tricky part, Select Euler A, a multiplier ( best results range between 0.4 to 0.6 ) and a Strengh, strengh is the total steps you will be applying the numpy file, if you go fot 36 steps a good number could range from 28 to 32 (from 70% to 90%).
Steps: you could have a good approach of the final image from 14-16 steps, and good results around 30 to 40 steps.
![re-generate](https://github.com/NeusZimmer/ONNX-ModularUI/assets/94193584/03afe051-ec35-438a-abcd-2e401f1bd4e6)
Guid: depends on the model, i found the need to increase it a little bit, 10 to 18 could be a good starting point (12-14 the usual)


