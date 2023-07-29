#from sched import scheduler
from Engine.General_parameters import Engine_Configuration
from Engine.pipelines_engines import Vae_and_Text_Encoders
from Engine.pipelines_engines import SchedulersConfig
#from own_pipes.pipeline_onnx_stable_diffusion_hires_txt2img import OnnxStableDiffusionHiResPipeline

import gc
#import numpy as np

"""from diffusers.utils import randn_tensor"""
#from diffusers.pipelines.stable_diffusion.pipeline_onnx_stable_diffusion_hires_txt2img import OnnxStableDiffusionHiResPipeline
from pipes.stable_diffusion.pipeline_onnx_stable_diffusion_hires_txt2img import OnnxStableDiffusionHiResPipeline



class Borg10:
    _shared_state = {}
    def __init__(self):
        self.__dict__ = self._shared_state

class txt2img_hires_pipe(Borg10):
    hires_pipe = None
    model = None
    seeds = []
    def __init__(self):
        Borg10.__init__(self)
    def __str__(self): return json.dumps(self.__dict__)

    def initialize(self,model_path,sched_name):
        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level=3
        sess_options.enable_cpu_mem_arena=False
        sess_options.enable_mem_reuse= True
        sess_options.enable_mem_pattern = True
        #sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

        if self.hires_pipe == None:
            #from Engine.General_parameters import Engine_Configuration as en_config
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

            self.hires_pipe = OnnxStableDiffusionHiResPipeline.from_pretrained(
                model_path,
                provider=provider,
                scheduler=SchedulersConfig().scheduler(sched_name,model_path),
                text_encoder=Vae_and_Text_Encoders().text_encoder,
                vae_decoder=Vae_and_Text_Encoders().vae_decoder,
                vae_encoder=Vae_and_Text_Encoders().vae_encoder,
                sess_options=sess_options
            )
        else:
            self.hires_pipe.scheduler=SchedulersConfig().scheduler(sched_name,model_path)


        import functools
        from Engine import lpw_pipe
        self.hires_pipe._encode_prompt = functools.partial(lpw_pipe._encode_prompt, self.hires_pipe)


        return self.hires_pipe

    def create_seeds(self,seed=None,iter=1,same_seeds=False):
        self.seeds=self.seed_generator(seed,iter)
        if same_seeds:
            for seed in self.seeds:
                seed = self.seeds[0]

    def unload_from_memory(self):
        self.hires_pipe= None
        self.model = None
        gc.collect()


    def seed_generator(self,seed,iteration_count):
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

    def run_inference(self,prompt,neg_prompt,hires_passes,height,width,hires_height,hires_width,steps,hires_steps,guid,eta,batch,seed):
    #def run_inference(self,prompt,neg_prompt,height,width,steps,guid,eta,batch,seed):
        import numpy as np
        rng = np.random.RandomState(seed)
        prompt.strip("\n")
        neg_prompt.strip("\n")

        lowres_image,hires_image = self.hires_pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            height=height,
            width=width,
            hires_height=hires_height,
            hires_width=hires_width,            
            num_inference_steps=steps,
            num_hires_steps=hires_steps,
            guidance_scale=guid,
            eta=eta,
            num_images_per_prompt=batch,
            prompt_embeds = None,
            negative_prompt_embeds = None,
            hires_steps=hires_passes,
            #callback= self.__callback,
            #callback_steps = running_config().Running_information["Callback_Steps"],
            #generator=rng).images
            generator=rng)

        dictio={'prompt':prompt,'neg_prompt':neg_prompt,'height':height,'width':width,'steps':steps,'guid':guid,'eta':eta,'batch':batch,'seed':seed}
        from Engine.General_parameters import running_config

        return lowres_image,hires_image,dictio