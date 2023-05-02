# ONNX-ModularUI
An approach to make a more modular UI for ONNX Stable Diffusion, to ease future upgrades

To make it work
Follow install steps from :https://github.com/Amblyopius/Stable-Diffusion-ONNX-FP16/ and use it as your virtual environment for this code.

Clone/Copy this repository on a directory of your preference, activate your python environment, and run "py ONNX-StableDiffusion

Current version provides support for: txt2img, img2img, Inpaint, instruct pix2pix.

Allows modification of the pipeline providers without re-running the UI., also, you may want to run some pipeline in one graphic card, another card for VAE and CPU to the rest... this UI allows such granularity for :main model, schedulers, VAE, text encoder...

Allows the use of a different VAE for a model (many models got the same VAE , then, why keep storing them on disk?

Add a clean memory option: changing resolution for the inferences keep garbage in memory, and it ends making an impact on the time needed for the inferences.

Also: wildcards, live prompts while running multiple iterations, a working deepdanbooru interrogator and resolution increase option. (one of my first tests with a ONNX model, a MS Model ([super-resolution-10.onnx]) to increase file resolution up to crazy sizes by slicing & joining (working , but not for professional uses)

For DeepDanbooru model, download it from: https://huggingface.co/Neus/Onnx_DeepDanbooru/tree/main

PD: first python code I did... sure it could be improved, but working fine and easy understand its workflows and to modify in future releases.


