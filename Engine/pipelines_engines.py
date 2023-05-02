from sched import scheduler
from Engine.General_parameters import Engine_Configuration
from Engine.pipeline_onnx_stable_diffusion_instruct_pix2pix import OnnxStableDiffusionInstructPix2PixPipeline
import gc
from diffusers import (
    OnnxRuntimeModel,
    OnnxStableDiffusionPipeline,
    OnnxStableDiffusionInpaintPipeline,
    OnnxStableDiffusionInpaintPipelineLegacy,
    OnnxStableDiffusionImg2ImgPipeline,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    DEISMultistepScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    UniPCMultistepScheduler,
# A partir de aqui, los extras de pruebas
    VQDiffusionScheduler,
    UnCLIPScheduler,
    KarrasVeScheduler,
    IPNDMScheduler,
    KDPM2AncestralDiscreteScheduler
)

class Borg:
    _shared_state = {}
    def __init__(self):
        self.__dict__ = self._shared_state


class SchedulersConfig(Borg):
    available_schedulers= None
    selected_scheduler= None
    def __init__(self):
        Borg.__init__(self)
        if self.available_schedulers == None:
            self.load_list()

    def __str__(self): return json.dumps(self.__dict__)

    def load_list(self):
        self.available_schedulers= ["DPMS_ms", "DPMS_ss", "EulerA", "Euler", "DDIM", "LMS", "PNDM", "DEIS", "HEUN", "KDPM2", "UniPC","KDPM2-A","Karras"]
        #self.available_schedulers= ["DPMS_ms", "DPMS_ss", "EulerA", "Euler", "DDIM", "LMS", "PNDM", "DEIS", "HEUN", "KDPM2", "UniPC","VQD","UnCLIP","Karras","KDPM2-A","IPNDMS"]
        #self.available_schedulers= ["DPMS_ms", "DPMS_ss", "EulerA", "Euler", "DDIM", "LMS", "PNDM", "DEIS", "HEUN", "KDPM2", "UniPC"]


    def scheduler(self,scheduler_name,model_path):
        scheduler = None
        provider = Engine_Configuration().Scheduler_provider
        match scheduler_name:
            case "PNDM":
                scheduler = PNDMScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)
            case "LMS":
                scheduler = LMSDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)
            case "DDIM" :
                scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)
            case "Euler" :
                scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)
            case "EulerA" :
                scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)
            case "DPMS_ms" :
                scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)
            case "DPMS_ss" :
                scheduler = DPMSolverSinglestepScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)   
            case "DEIS" :
                scheduler = DEISMultistepScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)
            case "HEUN" :
                scheduler = HeunDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)
            case "KDPM2":
                scheduler = KDPM2DiscreteScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)
            case "UniPC":
                scheduler = UniPCMultistepScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)  
#Test schedulers, maybe not working
            case "VQD":
                scheduler = VQDiffusionScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)  
            case "UnCLIP":
                scheduler = UnCLIPScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)  
            case "Karras":
                scheduler = KarrasVeScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)  
            case "KDPM2-A":
                scheduler = KDPM2AncestralDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)  
            case "IPNDMS":
                scheduler = IPNDMScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)  

        return scheduler


class Vae_and_Text_Encoders(Borg):
    vae_decoder = None
    vae_encoder = None
    text_encoder = None
    def __init__(self):
        Borg.__init__(self)

    def __str__(self): return json.dumps(self.__dict__)

    def load_vaedecoder(self,model_path):
        if " " in Engine_Configuration().VAEDec_provider:
            provider =eval(Engine_Configuration().VAEDec_provider)
        else:
            provider =Engine_Configuration().VAEDec_provider

        from Engine.General_parameters import UI_Configuration as UI_Configuration
        ui_config=UI_Configuration()
        if ui_config.Forced_VAE:
            vae_path=ui_config.forced_VAE_Dir
            print("Using Force VAE:"+vae_path)
        else:
            vae_path=model_path + "/vae_decoder"

        self.vae_decoder = OnnxRuntimeModel.from_pretrained(vae_path, provider=provider)
        return self.vae_decoder

    def load_vaeencoder(self,model_path):
        if " " in Engine_Configuration().VAEDec_provider:
            provider =eval(Engine_Configuration().VAEDec_provider)
        else:
            provider =Engine_Configuration().VAEDec_provider
        vae_path=model_path + "/vae_encoder"
        self.vae_encoder = OnnxRuntimeModel.from_pretrained(vae_path, provider=provider)
        return self.vae_encoder


    def load_textencoder(self,model_path):
        #Mirar si utilizar uno diferente (depende del tamaño en disco)
        if " " in Engine_Configuration().TEXTEnc_provider:
            provider = eval(Engine_Configuration().TEXTEnc_provider)
        else:
            provider = Engine_Configuration().TEXTEnc_provider
        self.text_encoder = OnnxRuntimeModel.from_pretrained(model_path + "/text_encoder", provider=provider)
        return self.text_encoder

    def unload_from_memory(self):
        self.vae_decoder = None
        self.vae_encoder = None
        self.text_encoder = None
        gc.collect()

class inpaint_pipe(Borg):
    inpaint_pipe = None
    model = None
    seeds = []
    def __init__(self):
        Borg.__init__(self)

    def __str__(self): return json.dumps(self.__dict__)

    def initialize(self,model_path,sched_name,legacy):
        from Engine.General_parameters import Engine_Configuration as en_config
        if Vae_and_Text_Encoders().text_encoder == None:
            Vae_and_Text_Encoders().load_textencoder(model_path)
        if Vae_and_Text_Encoders().vae_decoder == None:
            Vae_and_Text_Encoders().load_vaedecoder(model_path)
        if Vae_and_Text_Encoders().vae_encoder == None:
            Vae_and_Text_Encoders().load_vaeencoder(model_path)

        if " " in Engine_Configuration().MAINPipe_provider:
            provider =eval(Engine_Configuration().MAINPipe_provider)
        else:
            provider =Engine_Configuration().MAINPipe_provider

        if self.inpaint_pipe == None:
            if legacy:
                print("Legacy")
                self.inpaint_pipe = OnnxStableDiffusionInpaintPipelineLegacy.from_pretrained(
                    model_path,
                    provider=provider,
                    scheduler=SchedulersConfig().scheduler(sched_name,model_path),
                    text_encoder=Vae_and_Text_Encoders().text_encoder,
                    vae_decoder=Vae_and_Text_Encoders().vae_decoder,
                    vae_encoder=Vae_and_Text_Encoders().vae_encoder
                )
            else:
                print("No Legacy")
                self.inpaint_pipe = OnnxStableDiffusionInpaintPipeline.from_pretrained(
                    model_path,
                    provider=en_config().MAINPipe_provider,
                    scheduler=SchedulersConfig().scheduler(sched_name,model_path),
                    text_encoder=Vae_and_Text_Encoders().text_encoder,
                    vae_decoder=Vae_and_Text_Encoders().vae_decoder,
                    vae_encoder=Vae_and_Text_Encoders().vae_encoder
                )
        else:
             self.inpaint_pipe.scheduler=SchedulersConfig().scheduler(sched_name,model_path)
        return self.inpaint_pipe

    def create_seeds(self,seed=None,iter=1,same_seeds=False):
        self.seeds=seed_generator(seed,iter)
        if same_seeds:
            for seed in seeds:
                seed = seeds[0]

    def unload_from_memory(self):
        self.inpaint_pipe= None
        self.model = None
        #self.running = False
        gc.collect()


    def run_inference(self,prompt,neg_prompt,init_image,init_mask,height,width,steps,guid,eta,batch,seed,legacy):
        import numpy as np
        rng = np.random.RandomState(seed)
        prompt.strip("\n")
        neg_prompt.strip("\n")

        if legacy is True:
            batch_images = self.inpaint_pipe(
                prompt,
                negative_prompt=neg_prompt,
                image=init_image,
                mask_image=init_mask,
                num_inference_steps=steps,
                guidance_scale=guid,
                eta=eta,
                num_images_per_prompt=batch,
                generator=rng,
            ).images
        else:
            batch_images = self.inpaint_pipe(
                prompt,
                negative_prompt=neg_prompt,
                image=init_image,
                mask_image=init_mask,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guid,
                eta=eta,
                num_images_per_prompt=batch,
                generator=rng,
            ).images

        dictio={'prompt':prompt,'neg_prompt':neg_prompt,'height':height,'width':width,'steps':steps,'guid':guid,'eta':eta,'batch':batch,'seed':seed,'legacy':legacy}
        return batch_images,dictio



class txt2img_pipe(Borg):
    txt2img_pipe = None
    model = None
    running = False
    seeds = []
    def __init__(self):
        Borg.__init__(self)

    def __str__(self): return json.dumps(self.__dict__)

    def initialize(self,model_path,sched_name):
        from Engine.General_parameters import Engine_Configuration as en_config
        if Vae_and_Text_Encoders().text_encoder == None:
            Vae_and_Text_Encoders().load_textencoder(model_path)
            #Vae_and_Text_Encoders().load_textencoder(textenc_model_path)
        if Vae_and_Text_Encoders().vae_decoder == None:
            #Vae_and_Text_Encoders().load_vaedecoder(vae_model_path)
            Vae_and_Text_Encoders().load_vaedecoder(model_path)

        if " " in Engine_Configuration().MAINPipe_provider:
            provider =eval(Engine_Configuration().MAINPipe_provider)
        else:
            provider =Engine_Configuration().MAINPipe_provider

        if self.txt2img_pipe == None:
            self.txt2img_pipe = OnnxStableDiffusionPipeline.from_pretrained(
                model_path,
                provider=provider,
                scheduler=SchedulersConfig().scheduler(sched_name,model_path),
                text_encoder=Vae_and_Text_Encoders().text_encoder,
                vae_decoder=Vae_and_Text_Encoders().vae_decoder,
                vae_encoder=None
            )
        else:
             self.txt2img_pipe.scheduler=SchedulersConfig().scheduler(sched_name,model_path)
        return self.txt2img_pipe

    def create_seeds(self,seed=None,iter=1,same_seeds=False):
        self.seeds=seed_generator(seed,iter)
        if same_seeds:
            for seed in seeds:
                seed = seeds[0]



    def run_inference_test(self,prompt,neg_prompt,height,width,steps,guid,eta,batch,seed,image_np):
        import numpy as np
        image_np = np.reshape(image_np, (1,4,64,64))
        batch_images = self.txt2img_pipe(
            prompt,
            negative_prompt=neg_prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guid,
            eta=eta,
            num_images_per_prompt=batch,
            latents=image_np).images
        return batch_images, "vacio"

    def run_inference(self,prompt,neg_prompt,height,width,steps,guid,eta,batch,seed):
        #rng = np.random.RandomState(seeds[i])
        import numpy as np
        rng = np.random.RandomState(seed)
        prompt.strip("\n")
        neg_prompt.strip("\n")
        batch_images = self.txt2img_pipe(
            prompt,
            negative_prompt=neg_prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guid,
            eta=eta,
            num_images_per_prompt=batch,
            generator=rng).images
        #return batch_images,{'prompt':prompt,'neg_prompt':neg_prompt,'height':height,'width':width,'steps':steps,'guid':guid,'eta':eta,'batch':batch,'seed':seed}
        dictio={'prompt':prompt,'neg_prompt':neg_prompt,'height':height,'width':width,'steps':steps,'guid':guid,'eta':eta,'batch':batch,'seed':seed}
        return batch_images,dictio

    def unload_from_memory(self):
        self.txt2img_pipe= None
        self.model = None
        self.running = False
        gc.collect()


class instruct_p2p_pipe(Borg):
    instruct_p2p_pipe = None
    model = None
    seed = None

    def __init__(self):
        Borg.__init__(self)

    def __str__(self): return json.dumps(self.__dict__)

    def initialize(self,model_path,sched_name):
        from Engine.General_parameters import Engine_Configuration as en_config
        if Vae_and_Text_Encoders().text_encoder == None:
            Vae_and_Text_Encoders().load_textencoder(model_path)
        if Vae_and_Text_Encoders().vae_decoder == None:
            Vae_and_Text_Encoders().load_vaedecoder(model_path)
        if Vae_and_Text_Encoders().vae_encoder == None:
            Vae_and_Text_Encoders().load_vaeencoder(model_path)

        if " " in Engine_Configuration().MAINPipe_provider:
            provider =eval(Engine_Configuration().MAINPipe_provider)
        else:
            provider =Engine_Configuration().MAINPipe_provider

        if self.instruct_p2p_pipe == None:
            self.instruct_p2p_pipe = OnnxStableDiffusionInstructPix2PixPipeline.from_pretrained(
                model_path,
                provider=provider,
                scheduler=SchedulersConfig().scheduler(sched_name,model_path),
                text_encoder=Vae_and_Text_Encoders().text_encoder,
                vae_decoder=Vae_and_Text_Encoders().vae_decoder,
                vae_encoder=Vae_and_Text_Encoders().vae_encoder,
                safety_checker=None)
        else:
             self.instruct_p2p_pipe.scheduler=SchedulersConfig().scheduler(sched_name,model_path)

        return self.instruct_p2p_pipe


    def run_inference(self,prompt,input_image,steps,guid,eta):
        import numpy as np
        import torch
        prompt.strip("\n")
        generator = torch.Generator()
        generator = generator.manual_seed(self.seed)
        batch_images = self.instruct_p2p_pipe(
            prompt,
            image=input_image,
            num_inference_steps=steps,
            guidance_scale=guid,
            eta=eta,
            generator=generator,
            return_dict=False
        )
        dictio={'Pix2Pix prompt':prompt,'steps':steps,'guid':guid,'seed':self.seed}
        return batch_images[0],dictio

    def create_seed(self,seed=None):
        import numpy as np
        if seed == "" or seed == None:
            rng = np.random.default_rng()
            self.seed = int(rng.integers(np.iinfo(np.uint32).max))
        else:
            self.seed= int(seed)

    def unload_from_memory(self):
        self.instruct_p2p_pipe= None
        self.model = None
        gc.collect()


class img2img_pipe(Borg):
    img2img_pipe = None
    model = None
    seeds = []
    def __init__(self):
        Borg.__init__(self)

    def __str__(self): return json.dumps(self.__dict__)

    def initialize(self,model_path,sched_name):
        from Engine.General_parameters import Engine_Configuration as en_config
        if Vae_and_Text_Encoders().text_encoder == None:
            Vae_and_Text_Encoders().load_textencoder(model_path)
        if Vae_and_Text_Encoders().vae_decoder == None:
            Vae_and_Text_Encoders().load_vaedecoder(model_path)
        if Vae_and_Text_Encoders().vae_encoder == None:
            Vae_and_Text_Encoders().load_vaeencoder(model_path)

        if " " in Engine_Configuration().MAINPipe_provider:
            provider =eval(Engine_Configuration().MAINPipe_provider)
        else:
            provider =Engine_Configuration().MAINPipe_provider

        if self.img2img_pipe == None:
            self.img2img_pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(
                model_path,
                provider=provider,
                scheduler=SchedulersConfig().scheduler(sched_name,model_path),
                text_encoder=Vae_and_Text_Encoders().text_encoder,
                vae_decoder=Vae_and_Text_Encoders().vae_decoder,
                vae_encoder=Vae_and_Text_Encoders().vae_encoder
            )
        else:
             self.img2img_pipe.scheduler=SchedulersConfig().scheduler(sched_name,model_path)
        return self.img2img_pipe

    def create_seeds(self,seed=None,iter=1,same_seeds=False):
        self.seeds=seed_generator(seed,iter)
        if same_seeds:
            for seed in seeds:
                seed = seeds[0]

    def unload_from_memory(self):
        self.img2img_pipe= None
        self.model = None
        gc.collect()


    def run_inference(self,prompt,neg_prompt,init_image,strength,steps,guid,eta,batch,seed):
        import numpy as np
        rng = np.random.RandomState(seed)
        prompt.strip("\n")
        neg_prompt.strip("\n")

        batch_images = self.img2img_pipe(
            prompt,
            negative_prompt=neg_prompt,
            image=init_image,
            strength= strength,
            num_inference_steps=steps,
            guidance_scale=guid,
            eta=eta,
            num_images_per_prompt=batch,
            generator=rng,
        ).images
        dictio={'prompt':prompt,'neg_prompt':neg_prompt,'steps':steps,'guid':guid,'eta':eta,'strength':strength,'seed':seed}
        return batch_images,dictio


def seed_generator(seed,iteration_count):
    import numpy as np
    # generate seeds for iterations
    if seed == "" or seed == None:
        rng = np.random.default_rng()
        seed = rng.integers(np.iinfo(np.uint32).max)
    else:
        try:
            seed = int(seed) & np.iinfo(np.uint32).max
        except ValueError:
            seed = hash(seed) & np.iinfo(np.uint32).max

    # use given seed for the first iteration
    seeds = np.array([seed], dtype=np.uint32)

    if iteration_count > 1:
        seed_seq = np.random.SeedSequence(seed)
        seeds = np.concatenate((seeds, seed_seq.generate_state(iteration_count - 1)))

    return seeds