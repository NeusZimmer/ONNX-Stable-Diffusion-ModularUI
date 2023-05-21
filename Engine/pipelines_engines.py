from sched import scheduler
from Engine.General_parameters import Engine_Configuration
from Engine.pipeline_onnx_stable_diffusion_instruct_pix2pix import OnnxStableDiffusionInstructPix2PixPipeline
from Engine.pipeline_onnx_stable_diffusion_controlnet import OnnxStableDiffusionControlNetPipeline
import gc
import numpy as np

from diffusers.utils import randn_tensor
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

class Borg1:
    _shared_state = {}
    def __init__(self):
        self.__dict__ = self._shared_state
class Borg2:
    _shared_state = {}
    def __init__(self):
        self.__dict__ = self._shared_state
class Borg3:
    _shared_state = {}
    def __init__(self):
        self.__dict__ = self._shared_state
class Borg4:
    _shared_state = {}
    def __init__(self):
        self.__dict__ = self._shared_state
class Borg5:
    _shared_state = {}
    def __init__(self):
        self.__dict__ = self._shared_state
class Borg6:
    _shared_state = {}
    def __init__(self):
        self.__dict__ = self._shared_state

class SchedulersConfig(Borg):
    available_schedulers= None
    selected_scheduler= None
    _model_path = None
    _scheduler_name = None

    def __init__(self):
        Borg.__init__(self)
        if self.available_schedulers == None:
            self._load_list()

    def __str__(self): return json.dumps(self.__dict__)

    def _load_list(self):
        self.available_schedulers= ["DPMS_ms", "DPMS_ss", "EulerA", "Euler", "DDIM", "LMS", "PNDM", "DEIS", "HEUN", "KDPM2", "UniPC","KDPM2-A","Karras"]
        #self.available_schedulers= ["DPMS_ms", "DPMS_ss", "EulerA", "Euler", "DDIM", "LMS", "PNDM", "DEIS", "HEUN", "KDPM2", "UniPC","VQD","UnCLIP","Karras","KDPM2-A","IPNDMS"]
        #self.available_schedulers= ["DPMS_ms", "DPMS_ss", "EulerA", "Euler", "DDIM", "LMS", "PNDM", "DEIS", "HEUN", "KDPM2", "UniPC"]

    def reset_scheduler(self):
        return self.scheduler(self._scheduler_name,self._model_path)

    def scheduler(self,scheduler_name,model_path):
        scheduler = None
        self.selected_scheduler = None
        self._model_path = model_path
        self._scheduler_name = scheduler_name
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
        self.selected_scheduler =scheduler
        return self.selected_scheduler


class Vae_and_Text_Encoders(Borg1):
    vae_decoder = None
    vae_encoder = None
    text_encoder = None
    def __init__(self):
        Borg1.__init__(self)

    def __str__(self): return json.dumps(self.__dict__)

    def load_vaedecoder(self,model_path):
        from Engine.General_parameters import UI_Configuration as UI_Configuration

        if " " in Engine_Configuration().VAEDec_provider:
            provider =eval(Engine_Configuration().VAEDec_provider)
        else:
            provider =Engine_Configuration().VAEDec_provider

        ui_config=UI_Configuration()
        if ui_config.Forced_VAE:
            vae_path=ui_config.forced_VAE_Dir
            print("Using Forced VAE:"+vae_path)
        else:
            vae_path=model_path + "/vae_decoder"

        self.vae_decoder = None
        print(f"Loading VAE decoder in:{provider}" )
        self.vae_decoder = OnnxRuntimeModel.from_pretrained(vae_path, provider=provider)
        return self.vae_decoder

    def load_vaeencoder(self,model_path):
        if " " in Engine_Configuration().VAEDec_provider:
            provider =eval(Engine_Configuration().VAEDec_provider)
        else:
            provider =Engine_Configuration().VAEDec_provider

        vae_path=model_path + "/vae_encoder"
        self.vae_encoder = None
        print(f"Loading VAE encoder in:{provider}" )
        self.vae_encoder = OnnxRuntimeModel.from_pretrained(vae_path, provider=provider)
        return self.vae_encoder

    def load_textencoder(self,model_path):
        #Mirar si utilizar uno diferente (depende del tamaño en disco)
        if " " in Engine_Configuration().TEXTEnc_provider:
            provider = eval(Engine_Configuration().TEXTEnc_provider)
        else:
            provider = Engine_Configuration().TEXTEnc_provider

        print(f"Loading TEXT encoder in:{provider}" )
        self.text_encoder = None
        self.text_encoder = OnnxRuntimeModel.from_pretrained(model_path + "/text_encoder", provider=provider)
        return self.text_encoder

    def unload_from_memory(self):
        self.vae_decoder = None
        self.vae_encoder = None
        self.text_encoder = None
        gc.collect()

class inpaint_pipe(Borg2):
    inpaint_pipe = None
    model = None
    seeds = []
    def __init__(self):
        Borg2.__init__(self)

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



class txt2img_pipe(Borg3):
    txt2img_pipe = None
    model = None
    running = False
    seeds = []
    latents_list = []

    def __init__(self):
        Borg3.__init__(self)
        self.latents_list = []

    def __str__(self): return json.dumps(self.__dict__)

    def initialize(self,model_path,sched_name):
        from Engine.General_parameters import Engine_Configuration as en_config
        if Vae_and_Text_Encoders().text_encoder == None:
            Vae_and_Text_Encoders().load_textencoder(model_path)
        if Vae_and_Text_Encoders().vae_decoder == None:
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

        import functools
        from Engine import lpw_pipe
        self.txt2img_pipe._encode_prompt = functools.partial(lpw_pipe._encode_prompt, self.txt2img_pipe)
        from Engine import txt2img_pipe_sub
        self.txt2img_pipe.__call__ = functools.partial(txt2img_pipe_sub.__call__, self.txt2img_pipe)
        OnnxStableDiffusionPipeline.__call__ =  txt2img_pipe_sub.__call__

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


    def get_ordered_latents(self):
        from Engine.General_parameters import running_config
        import numpy as np
        name=running_config().Running_information["Latent_Name"]
        name1= name.split(',')
        lista=[0]*len(name1)
        for pair in name1:
            tupla= pair.split(':')
            lista[int(tupla[0])-1]=tupla[1]
        print("Ordered numpys")
        print(lista)
        return lista


    def sum_latents(self,latent_list,formula):
        loaded_latent= None

        #print("Analizando la formula:"+str(formula))

        subformula_startmarks=list([pos for pos, char in enumerate(formula) if char == '('])
        subformula_endmarks=list([pos for pos, char in enumerate(formula) if char == ')'])

        if (len(subformula_endmarks) != len(subformula_startmarks)):
            raise Exception("Sorry, Error in formula, check it")

        if (len(subformula_startmarks) > 0):
            contador1=0
            while (len(subformula_startmarks)>contador1) and (subformula_startmarks[contador1] < subformula_endmarks[0]):
                contador1+=1

            #Extracts the first subformula
            subformula=formula[(subformula_startmarks[0]+1):subformula_endmarks[contador1-1]]

            previo=formula[0:(subformula_startmarks[0])]
            posterior=formula[subformula_endmarks[contador1-1]+1:]

            retorno, retorno2=self.sum_latents(latent_list,subformula)

            if (len(previo)>0):
                latent_previo, retorno_previo=self.sum_latents(latent_list,previo[:-1])
                if ((previo[-1] =='w') or (previo[-1] =='W')):
                    retorno = latent_previo + " Horizontal " + retorno
                    retorno2 = np.concatenate((retorno_previo,retorno2),axis=3) #left & right
                else:
                    retorno = latent_previo + " Vertical " + retorno
                    retorno2 = np.concatenate((retorno_previo,retorno2),axis=2)  #Up & Down

            if (len(posterior)>0):
                latent_posterior, retorno_posterior = self.sum_latents(latent_list,posterior[1:])
                if ((posterior[0] =='w') or (posterior[0] =='W')):
                    retorno = retorno + " Horizontal " + latent_posterior
                    retorno2 = np.concatenate((retorno,retorno_posterior),axis=3) #left & right
                else:
                    retorno = retorno + " Vertical " + latent_posterior
                    retorno2 = np.concatenate((retorno,retorno_posterior),axis=2)  #Up & Down
        else:
            position=-1
            for pos, char in enumerate(formula):
                if char in "WwHh":
                    position=pos
                    break
            if position ==-1:
                retorno=formula
                retorno2=self.load_latent_file(latent_list,formula)
            else:
                index=formula[0:position]
                retorno2=self.load_latent_file(latent_list,formula[0:position])
                resultado3,retorno3=self.sum_latents(latent_list,formula[position+1:])
                if (formula[position]=='w') or (formula[position] =='W'):
                    retorno=index + "horizontal" + resultado3
                    retorno2 = np.concatenate((retorno2,retorno3),axis=3) #left & right
                else:
                    retorno=index + "vertical" + resultado3
                    retorno2 = np.concatenate((retorno2,retorno3),axis=2)  #Up & Down

        return retorno, retorno2

    def load_latent_file(self,latent_list,index):
        loaded_latent = None
        if True:
        #try:
            name=latent_list[int(index)-1]
            print(f"Loading latent(idx:name):{index}:{name}")
            loaded_latent=np.load(f"./latents/{name}")
            print("Latent loaded")
        #except:
            #print("Latent not found")  #if not found, just create noise, size ¿? 
        return loaded_latent



    def get_initial_latent(self, steps,multiplier,generator,strengh):
        from Engine.General_parameters import running_config
        latent_list=self.get_ordered_latents()
        formula=running_config().Running_information["Latent_Formula"]
        formula=formula.replace(' ', '')
        formulafinal,loaded_latent=self.sum_latents(latent_list,formula)
        print("Formula final"+formulafinal)

        print("Resultant Latent Shape")
        print("H:"+str(loaded_latent.shape[2]*8)+"x W:"+str(loaded_latent.shape[3]*8))

        self.txt2img_pipe.scheduler = SchedulersConfig().reset_scheduler()

        if multiplier < 1:
            print("Multiplier applied (Use 1 as value, to do not apply)")
            loaded_latent= multiplier * loaded_latent

        noise = (0.1825 * generator.random(loaded_latent.shape) + 0.3).astype(loaded_latent.dtype) #works a lot better for EulerA than other schedulers  , why?

        #if True:
                #loaded_latent2= np.concatenate((loaded_latent2,loaded_latent1),axis=2)  #primero arriba y abajo
                #loaded_latent= np.concatenate((loaded_latent,loaded_latent2),axis=3) #luego  + drcha+izda
                #noise = generator.randn(*loaded_latent.shape).astype(loaded_latent.dtype)
                #noise = (multiplier * np.random.random(loaded_latent.shape) + 0.3).astype(loaded_latent.dtype)
                #noise = np.random.Generator.random(loaded_latent.shape,loaded_latent.dtype) 


        offset = self.txt2img_pipe.scheduler.config.get("steps_offset", 0)
        #init_timestep = int(steps * strengh) + offset #Con 0.ocho funciona, con 9 un poco peor?, probar
        init_timestep = int(steps * strengh) + 0 #Con 0.ocho funciona, con 9 un poco peor?, probar
        init_timestep = min(init_timestep, steps)
        #init_timestep = strengh

        timesteps = self.txt2img_pipe.scheduler.timesteps.numpy()[-init_timestep]
        #timesteps = np.array([timesteps] * batch_size * num_images_per_prompt)


        import torch
        init_latents = self.txt2img_pipe.scheduler.add_noise(
            torch.from_numpy(loaded_latent), (torch.from_numpy(noise)).type(torch.LongTensor), (torch.from_numpy(np.array([timesteps])).type(torch.LongTensor))
        )
        init_latents = init_latents.numpy()

        return init_latents
        #return loaded_latent


    def run_inference(self,prompt,neg_prompt,height,width,steps,guid,eta,batch,seed,multiplier,strengh):
        import numpy as np
        rng = np.random.RandomState(seed)
        prompt.strip("\n")
        neg_prompt.strip("\n")
        loaded_latent= None
        from Engine.General_parameters import running_config

        if running_config().Running_information["Load_Latents"]:
            loaded_latent=self.get_initial_latent(steps,multiplier,rng,strengh)

        batch_images = self.txt2img_pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guid,
            eta=eta,
            num_images_per_prompt=batch,
            prompt_embeds = None,
            negative_prompt_embeds = None,
            latents=loaded_latent,
            callback= self.__callback,
            callback_steps = running_config().Running_information["Callback_Steps"],
            generator=rng).images

        dictio={'prompt':prompt,'neg_prompt':neg_prompt,'height':height,'width':width,'steps':steps,'guid':guid,'eta':eta,'batch':batch,'seed':seed,'strengh':strengh}
        from Engine.General_parameters import running_config
        if running_config().Running_information["Save_Latents"]:
            print("Saving all latent_steps to disk")
            self.savelatents_todisk(seed=seed,contador=len(self.latents_list))
            print("Latents Saved")
        return batch_images,dictio


    def savelatents_todisk(self,path="./latents",seed=0,save_steps=False,contador=1000,callback_steps=2):
        import numpy as np
        if self.latents_list:
            latent_to_save= self.latents_list.pop()
            if save_steps:
                self.savelatents_todisk(path=path,seed=seed,save_steps=save_steps,contador=contador-1,callback_steps=callback_steps)
            np.save(f"{path}/Seed-{seed}_latent_Step-{contador*callback_steps}.npy", latent_to_save)
        return


    def __callback(self,i, t, latents):
        from Engine.General_parameters import running_config
        cancel = running_config().Running_information["cancelled"]
        if running_config().Running_information["Save_Latents"]:
            self.latents_list.append(latents)
        return  cancel

    def unload_from_memory(self):
        self.txt2img_pipe= None
        self.model = None
        self.running = False
        self.latents_list = None
        gc.collect()



class instruct_p2p_pipe(Borg4):
    instruct_p2p_pipe = None
    model = None
    seed = None

    def __init__(self):
        Borg4.__init__(self)

    def __str__(self): return json.dumps(self.__dict__)

    def initialize(self,model_path,sched_name):
        from Engine.General_parameters import Engine_Configuration as en_config
        if Vae_and_Text_Encoders().text_encoder == None:
            Vae_and_Text_Encoders().load_textencoder(model_path)
        if Vae_and_Text_Encoders().vae_decoder == None:
            Vae_and_Text_Encoders().load_vaedecoder(model_path)
        if Vae_and_Text_Encoders().vae_encoder == None:
            Vae_and_Text_Encoders().load_vaeencoder(model_path)

        print(Vae_and_Text_Encoders().text_encoder)
        print(Vae_and_Text_Encoders().vae_decoder)
        print(Vae_and_Text_Encoders().vae_encoder)

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


class img2img_pipe(Borg5):
    img2img_pipe = None
    model = None
    seeds = []
    def __init__(self):
        Borg5.__init__(self)

    def __str__(self): return json.dumps(self.__dict__)

    def initialize(self,model_path,sched_name):
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


class ControlNet_pipe(Borg6):
#import onnxruntime as ort
    controlnet_Model_ort= None
    #controlnet_unet_ort= None
    ControlNet_pipe = None
    seeds = []

    def __init__(self):
        Borg6.__init__(self)

    def __str__(self): return json.dumps(self.__dict__)

    def __load_ControlNet_model(self,model_path,ControlNET_drop):

        if " " in Engine_Configuration().ControlNet_provider:
            provider =eval(Engine_Configuration().ControlNet_provider)
        else:
            provider =Engine_Configuration().ControlNet_provider
        print("Loading Controlnet:"+str(provider))
        from Engine.General_parameters import ControlNet_config
        available_models=dict(ControlNet_config().available_controlnet_models())
        print(available_models)
        ControlNet_path = available_models[ControlNET_drop]
        print(ControlNet_path)

        controlnet_model = OnnxRuntimeModel.from_pretrained(ControlNet_path, provider=provider)
        return controlnet_model

    def __load_uNet_model(self,model_path):
        #Aqui cargar con ort el modelo unicamente en el provider principal.
        print("Cargando Unet")
        if " " in Engine_Configuration().MAINPipe_provider:
            provider =eval(Engine_Configuration().MAINPipe_provider)
        else:
            provider =Engine_Configuration().MAINPipe_provider

        unet_model = OnnxRuntimeModel.from_pretrained(model_path + "/unet", provider=provider)
        return unet_model

    def initialize(self,model_path,sched_name,ControlNET_drop):
        #from Engine.General_parameters import Engine_Configuration as en_config
        if Vae_and_Text_Encoders().text_encoder == None:
            Vae_and_Text_Encoders().load_textencoder(model_path)
        if Vae_and_Text_Encoders().vae_decoder == None:
            Vae_and_Text_Encoders().load_vaedecoder(model_path)
        if Vae_and_Text_Encoders().vae_encoder == None:
            Vae_and_Text_Encoders().load_vaeencoder(model_path)

        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
    
        if self.ControlNet_pipe == None:
            print("LLamando a las cargas")
            self.controlnet_Model_ort= self.__load_ControlNet_model(model_path,ControlNET_drop)
            #self.controlnet_unet_ort= self.__load_uNet_model(model_path)
            print("Creando pipe")
            self.ControlNet_pipe = OnnxStableDiffusionControlNetPipeline.from_pretrained(
                #unet=self.controlnet_unet_ort,
                model_path,
                controlnet=self.controlnet_Model_ort,
                vae_encoder=Vae_and_Text_Encoders().vae_encoder,
                vae_decoder=Vae_and_Text_Encoders().vae_decoder,
                text_encoder=Vae_and_Text_Encoders().text_encoder,
                scheduler=SchedulersConfig().scheduler(sched_name,model_path),
                #sess_options=opts, 
                provider = Engine_Configuration().MAINPipe_provider,
                requires_safety_checker= False
            )
        return self.ControlNet_pipe

    def run_inference(self,prompt,neg_prompt,input_image,pose_image,width,height,eta,steps,guid,seed):
        import numpy as np
        print(seed)
        rng = np.random.RandomState(int(seed))
        image = self.ControlNet_pipe(
            prompt,
            pose_image,
            negative_prompt=neg_prompt,
            width = width,
            height = height,
            num_inference_steps = steps,
            guidance_scale=guid,
            eta=eta,
            num_images_per_prompt=1,
            generator=rng,
        ).images[0]
        #Añadir el diccionario
        return image

    def create_seeds(self,seed=None,iter=1,same_seeds=False):
        self.seeds=seed_generator(seed,iter)
        if same_seeds:
            for seed in seeds:
                seed = seeds[0]

    def unload_from_memory(self):
        self.ControlNet_pipe= None
        controlnet_Model_ort= None
        #controlnet_unet_ort= None
        gc.collect()


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