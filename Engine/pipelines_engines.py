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
# Non working schedulers
    VQDiffusionScheduler,
    UnCLIPScheduler,
    KarrasVeScheduler,
    IPNDMScheduler,
    KDPM2AncestralDiscreteScheduler,
    DDIMInverseScheduler,
    ScoreSdeVeScheduler
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
    _low_res_scheduler = None

    def __init__(self):
        Borg.__init__(self)
        if self.available_schedulers == None:
            self._load_list()

    def __str__(self): return json.dumps(self.__dict__)

    def _load_list(self):
        self.available_schedulers= ["DPMS_ms", "DPMS_ss", "DPMS++_Heun","DPMS_Heun", "EulerA", "Euler", "DDIM", "LMS", "PNDM", "DEIS", "HEUN", "KDPM2", "UniPC","KDPM2-A"]
        #self.available_schedulers= ["DPMS_ms", "DPMS_ss", "EulerA", "Euler", "DDIM", "LMS", "PNDM", "DEIS", "HEUN", "KDPM2", "UniPC","VQD","UnCLIP","Karras","KDPM2-A","IPNDMS","DDIM-Inverse","SDE-1"]
        #self.available_schedulers= ["DPMS_ms", "DPMS_ss", "EulerA", "Euler", "DDIM", "LMS", "PNDM", "DEIS", "HEUN", "KDPM2", "UniPC"]

    def schedulers_controlnet_list(self):
        return ["DPMS_ms", "DPMS_ss", "DPMS++_Heun","DPMS_Heun", "DDIM", "LMS", "PNDM"]

    def reset_scheduler(self):
        return self.scheduler(self._scheduler_name,self._model_path)
    
    def low_res_scheduler(self,model_path=None):
        if model_path==None:
            model_path=self._model_path
        self._low_res_scheduler = DPMSolverSinglestepScheduler.from_pretrained(self._model_path, subfolder="scheduler",provider=['DmlExecutionProvider'])
        return self._low_res_scheduler    

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
            case "DDIM-Inverse":
                scheduler = DDIMInverseScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)  
            case "DPMS_Heun":
                scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider,algorithm_type="dpmsolver", solver_type="heun") 
            case "DPMS++_Heun":
                scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider,algorithm_type="dpmsolver++", solver_type="heun") 
            case "SDE-1":
                scheduler = ScoreSdeVeScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider,algorithm_type="dpmsolver++", solver_type="heun") 
           
            
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
        from Engine.General_parameters import running_config

        if " " in Engine_Configuration().VAEDec_provider:
            provider =eval(Engine_Configuration().VAEDec_provider)
        else:
            provider =Engine_Configuration().VAEDec_provider

        running_config=running_config()
        import os
        if running_config.Running_information["Vae_Config"]:
            vae_config=running_config.Running_information["Vae_Config"]
            vae_path1= (model_path + "/vae_decoder") if vae_config[0]=="model" else vae_config[0]
            vae_path2= (model_path + "/vae_decoder") if vae_config[1]=="model" else vae_config[1]
            vae_path3= (model_path + "/vae_decoder") if vae_config[2]=="model" else vae_config[2]                       
            vae_path=""

            if os.path.exists(vae_path1): vae_path= vae_path1
            elif os.path.exists(vae_path2): vae_path= vae_path2
            elif os.path.exists(vae_path3): vae_path= vae_path3
            else: raise Exception("No valid vae decoder path"+vae_path)

        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level=3

        self.vae_decoder = None
        print(f"Loading VAE decoder in:{provider}, from {vae_path}" )
        self.vae_decoder = OnnxRuntimeModel.from_pretrained(vae_path, provider=provider,sess_options=sess_options)
        return self.vae_decoder

    def load_vaeencoder(self,model_path):
        from Engine.General_parameters import running_config
        running_config=running_config()
        import os

        vae_config=running_config.Running_information["Vae_Config"]
        vae_path1= (model_path + "/vae_encoder") if vae_config[3]=="model" else vae_config[3]
        vae_path2= (model_path + "/vae_encoder") if vae_config[4]=="model" else vae_config[4]
        vae_path3= (model_path + "/vae_encoder") if vae_config[5]=="model" else vae_config[5]
        vae_path=""

        if os.path.exists(vae_path1): vae_path= vae_path1
        elif os.path.exists(vae_path2): vae_path= vae_path2
        elif os.path.exists(vae_path3): vae_path= vae_path3
        else: raise Exception("No valid vae encoder path:"+vae_path)


        if " " in Engine_Configuration().VAEDec_provider:
            provider =eval(Engine_Configuration().VAEDec_provider)
        else:
            provider =Engine_Configuration().VAEDec_provider

        #vae_path=model_path + "/vae_encoder"
        self.vae_encoder = None

        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level=3
        print(f"Loading VAE encoder in:{provider}, from {vae_path}" )
        #self.vae_encoder = OnnxRuntimeModel.from_pretrained(vae_path, provider='DmlExecutionProvider',sess_options=sess_options)
        self.vae_encoder = OnnxRuntimeModel.from_pretrained(vae_path, provider='provider',sess_options=sess_options)
        #print("Acordarse de cambiar donde carga el vae_encoder")

        return self.vae_encoder

    def load_textencoder(self,model_path):
        from Engine.General_parameters import running_config

        if " " in Engine_Configuration().TEXTEnc_provider:
            provider = eval(Engine_Configuration().TEXTEnc_provider)
        else:
            provider = Engine_Configuration().TEXTEnc_provider

        running_config=running_config()
        import os
        if running_config.Running_information["Textenc_Config"]:
            Textenc_Config=running_config.Running_information["Textenc_Config"]
            Textenc_path1= (model_path + "/text_encoder") if Textenc_Config[0]=="model" else Textenc_Config[0]
            Textenc_path2= (model_path + "/text_encoder") if Textenc_Config[1]=="model" else Textenc_Config[1]                     
            Textenc_path=""
            if os.path.exists(Textenc_path1): Textenc_path= Textenc_path1
            elif os.path.exists(Textenc_path2): Textenc_path= Textenc_path2
            else: raise Exception("No valid Text Encoder path:"+Textenc_path)


        print(f"Loading TEXT encoder in:{provider} from:{Textenc_path}" )
        self.text_encoder = None
        #self.text_encoder = OnnxRuntimeModel.from_pretrained(model_path + "/text_encoder", provider=provider)
        self.text_encoder = OnnxRuntimeModel.from_pretrained(Textenc_path, provider=provider)
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

        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level=3

        if self.inpaint_pipe == None:
            if legacy:
                print("Legacy")
                print(f"Loading Inpaint unet pipe in {provider}")
                self.inpaint_pipe = OnnxStableDiffusionInpaintPipelineLegacy.from_pretrained(
                    model_path,
                    provider=provider,
                    scheduler=SchedulersConfig().scheduler(sched_name,model_path),
                    text_encoder=Vae_and_Text_Encoders().text_encoder,
                    vae_decoder=Vae_and_Text_Encoders().vae_decoder,
                    vae_encoder=Vae_and_Text_Encoders().vae_encoder,
                    sess_options=sess_options                    
                )
            else:
                print("No Legacy")
                print(f"Loading Inpaint unet pipe in {provider}")
                self.inpaint_pipe = OnnxStableDiffusionInpaintPipeline.from_pretrained(
                    model_path,
                    provider=en_config().MAINPipe_provider,
                    scheduler=SchedulersConfig().scheduler(sched_name,model_path),
                    text_encoder=Vae_and_Text_Encoders().text_encoder,
                    vae_decoder=Vae_and_Text_Encoders().vae_decoder,
                    vae_encoder=Vae_and_Text_Encoders().vae_encoder,
                    sess_options=sess_options
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

    def reinitialize(self,model_path):
        from Engine.General_parameters import Engine_Configuration as en_config

        if " " in Engine_Configuration().MAINPipe_provider:
            provider =eval(Engine_Configuration().MAINPipe_provider)
        else:
            provider =Engine_Configuration().MAINPipe_provider


        unet_path=model_path+"/unet"
        self.txt2img_pipe.unet = OnnxRuntimeModel.from_pretrained(unet_path,provider=provider)

        import functools
        from Engine import lpw_pipe
        self.txt2img_pipe._encode_prompt = functools.partial(lpw_pipe._encode_prompt, self.txt2img_pipe)
        from Engine import txt2img_pipe_sub
        self.txt2img_pipe.__call__ = functools.partial(txt2img_pipe_sub.__call__, self.txt2img_pipe)
        OnnxStableDiffusionPipeline.__call__ =  txt2img_pipe_sub.__call__

        return self.txt2img_pipe.unet



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
            import onnxruntime as ort
            #from optimum.onnxruntime import ORTStableDiffusionPipeline as ort
            sess_options = ort.SessionOptions()
            sess_options.log_severity_level=3
            print(f"Loadint Txt2Img Pipeline in [{provider}]")            
            self.txt2img_pipe = OnnxStableDiffusionPipeline.from_pretrained(
                model_path,
                provider=provider,
                scheduler=SchedulersConfig().scheduler(sched_name,model_path),
                text_encoder=Vae_and_Text_Encoders().text_encoder,
                vae_decoder=Vae_and_Text_Encoders().vae_decoder,
                vae_encoder=None,
                sess_options=sess_options
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
        #print("Ordered numpys"+str(lista))
        return lista

    def sum_latents(self,latent_list,formula,generator,resultant_latents,iter=0):
        #print("Processing formula:"+str(formula))
        subformula_latents= None
        while ("(" in formula) or (")" in formula):
            #print("Subformula exists")
            subformula_startmarks=list([pos for pos, char in enumerate(formula) if char == '('])
            subformula_endmarks=list([pos for pos, char in enumerate(formula) if char == ')'])

            if (len(subformula_endmarks) != len(subformula_startmarks)):
                raise Exception("Sorry, Error in formula, check it")

            contador=0
            while (len(subformula_startmarks)>contador) and (subformula_startmarks[contador] < subformula_endmarks[0]):
                contador+=1
            if contador==0: raise Exception("Sorry, Error in formula, check it")

            subformula= formula[(subformula_startmarks[contador-1]+1):subformula_endmarks[0]]
            #print(f"subformula:{iter},{subformula}")
            previous= formula[0:subformula_startmarks[contador-1]]
            posterior=formula[subformula_endmarks[0]+1:]
            formula= f"{previous}|{iter}|{posterior}" 
            iter+=1
            subformula_latents =  self.sum_latents(latent_list,subformula,generator,resultant_latents,iter)
            resultant_latents.append(subformula_latents)


        # Here we got a plain formula
        #print("No subformulas")
        result = self.process_simple_formula(latent_list,formula,generator,resultant_latents)
        return result

    def process_simple_formula(self,latent_list,formula,generator,resultant_latents):
        position=-1
        #print("Simple_formula process")
        for pos, char in enumerate(formula):
            if char in "WwHh":
                position=pos
                break
        if position ==-1 and len(formula)>0:  #No operators, single item
            result=self.load_latent_file(latent_list,formula,generator,resultant_latents)
        else:
            previous=formula[0:position]
            operator=formula[position]
            rest=formula[position+1:]
            #print("previous:"+previous)
            #print("operator:"+operator)
            #print("rest:"+rest)

            result=self.load_latent_file(latent_list,previous,generator,resultant_latents)
            result2 = self.process_simple_formula(latent_list,rest,generator,resultant_latents)

            if (operator=='w'):
                result = self._sum_latents(result,result2,True) #left & right
            elif (operator=='h'):
                result = self._sum_latents(result,result2,False) #Up & Down

        return result


    def load_latent_file(self,latent_list,data,generator,resultant_latents):
        result = ""
        if "|" in data:
            lista=data.split("|")
            index=int(lista[1])
            result = resultant_latents[index]
            #result = "SP:"+resultant_latents[index]
        else:
            index=int(data)
            name=latent_list[int(index)-1]
            if "noise" not in name:
                print(f"Loading latent(idx:name):{index}:{name}")
                result=np.load(f"./latents/{name}")
                if False:
                    print("Multiplier 0.18215 applied")
                    loaded_latent= 0.18215 * result
            else:
                noise_size=name.split("noise-")[1].split("x")
                print(f"Creating noise block of W/H:{noise_size}")
                noise = (0.3)*(generator.random((1,4,int(int(noise_size[1])/8),int(int(noise_size[0])/8))).astype(np.float32))
                #noise = (generator.random((1,4,int(int(noise_size[1])/8),int(int(noise_size[0])/8))).astype(np.float32))
                result = noise

        return result



    def _sum_latents(self,latent1,latent2,direction): #direction True=horizontal sum(width), False=vertical sum(height)
        latent_sum= None
        side=""
        try:
            if direction:
                side="Height"
                latent_sum = np.concatenate((latent1,latent2),axis=3) #left & right
            else:
                side="Width"
                latent_sum = np.concatenate((latent1,latent2),axis=2)  #Up & Down
        except:
            size1=f"Latent1={(latent1.shape[3]*8)}x{(latent1.shape[2]*8)}"
            size2=f"Latent2={(latent2.shape[3]*8)}x{(latent2.shape[2]*8)}"
            raise Exception(f"Cannot sum the latents(Width x Height):{size1} and {size2} its {side} must be equal")
        return latent_sum


    def get_initial_latent(self, steps,multiplier,generator,strengh):
        debug = False
        from Engine.General_parameters import running_config
        latent_list=self.get_ordered_latents()
        formula=running_config().Running_information["Latent_Formula"]
        formula=formula.replace(' ', '')
        formula=formula.lower()
        #formulafinal,loaded_latent=self.sum_latents(latent_list,formula,generator,[])
        #print("Formula final"+formulafinal)
        loaded_latent=self.sum_latents(latent_list,formula,generator,[])

        print("Resultant Latent Shape "+"H:"+str(loaded_latent.shape[2]*8)+"x W:"+str(loaded_latent.shape[3]*8))

        self.txt2img_pipe.scheduler = SchedulersConfig().reset_scheduler()
        if multiplier < 1:
            print("Multiplier applied (Use 1 as value, to do not apply)")
            loaded_latent= multiplier * loaded_latent

        noise = (0.3825 * generator.random(loaded_latent.shape)).astype(loaded_latent.dtype) #works a lot better for EulerA&DDIM than other schedulers  , why?
        #noise = (0.1825 * generator.random(loaded_latent.shape) + 0.3).astype(loaded_latent.dtype) #works a lot better for EulerA&DDIM than other schedulers  , why?
        #noise = (generator.random(loaded_latent.shape)).astype(loaded_latent.dtype)

        offset = self.txt2img_pipe.scheduler.config.get("steps_offset", 0)
        if True:
            offset= running_config().Running_information["offset"]
        print(f"Offset:{offset}")
        #init_timestep = int(steps * strengh) + offset #Con 0.ocho funciona, con 9 un poco peor?, probar
        init_timestep = int(steps * strengh) - offset #Con 0.ocho funciona, con 9 un poco peor?, probar, aqui tenia puesto offset a 0
        print(f"init_timestep, {init_timestep}")
        init_timestep = min(init_timestep, steps)
        print(f"init_timestep, {init_timestep}")
        timesteps = self.txt2img_pipe.scheduler.timesteps.numpy()[-init_timestep]
        #timesteps = self.txt2img_pipe.scheduler.timesteps.numpy()[-offset]
        print(f"timesteps, {timesteps}")
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

        #self.txt2img_pipe.load_textual_inversion("./Engine/test.pt", token="tester")

        if running_config().Running_information["Load_Latents"]:
            loaded_latent=self.get_initial_latent(steps,multiplier,rng,strengh)
        prompt_embeds0 = None
        """compel=False

        if compel:

            from compel import Compel
            compel = Compel(tokenizer=self.txt2img_pipe.tokenizer, text_encoder=self.txt2img_pipe.text_encoder)
            prompt_embeds0=compel(prompt)
            print(prompt_embeds0)
            print(type(prompt_embeds0))
            prompt=None"""


        batch_images = self.txt2img_pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guid,
            eta=eta,
            num_images_per_prompt=batch,
            prompt_embeds = prompt_embeds0,
            negative_prompt_embeds = None,
            latents=loaded_latent,
            callback= self.__callback,
            callback_steps = running_config().Running_information["Callback_Steps"],
            generator=rng).images

        dictio={'prompt':prompt,'neg_prompt':neg_prompt,'height':height,'width':width,'steps':steps,'guid':guid,'eta':eta,'batch':batch,'seed':seed,'strengh':strengh}
        from Engine.General_parameters import running_config
        if running_config().Running_information["Save_Latents"]:
            print("Saving last latent steps to disk")
            self.savelatents_todisk(seed=seed,contador=len(self.latents_list))
            #print("Latents Saved")
        return batch_images,dictio


    def savelatents_todisk(self,path="./latents",seed=0,save_steps=True,contador=1000,callback_steps=2):
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

        if " " in Engine_Configuration().MAINPipe_provider:
            provider =eval(Engine_Configuration().MAINPipe_provider)
        else:
            provider =Engine_Configuration().MAINPipe_provider

        if self.instruct_p2p_pipe == None:
            print(f"Loading Instruct pix2pix pipe in {provider}")
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

        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level=3


        if self.img2img_pipe == None:
            print(f"Loading Img2Img pipe in {provider}")
            self.img2img_pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(
                model_path,
                provider=provider,
                scheduler=SchedulersConfig().scheduler(sched_name,model_path),
                text_encoder=Vae_and_Text_Encoders().text_encoder,
                vae_decoder=Vae_and_Text_Encoders().vae_decoder,
                vae_encoder=Vae_and_Text_Encoders().vae_encoder,
                sess_options=sess_options
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
        dictio={'Img2ImgPrompt':prompt,'neg_prompt':neg_prompt,'steps':steps,'guid':guid,'eta':eta,'strength':strength,'seed':seed}
        return batch_images,dictio


class ControlNet_pipe(Borg6):
#import onnxruntime as ort
    controlnet_Model_ort= None
    #controlnet_unet_ort= None
    ControlNET_Name=None
    ControlNet_pipe = None
    seeds = []

    def __init__(self):
        Borg6.__init__(self)

    def __str__(self): return json.dumps(self.__dict__)

    def load_ControlNet_model(self,model_path,ControlNET_drop):
        self.__load_ControlNet_model(model_path,ControlNET_drop)

    def __load_ControlNet_model(self,model_path,ControlNET_drop):
        from Engine.General_parameters import ControlNet_config
        import onnxruntime as ort

        if " " in Engine_Configuration().ControlNet_provider:
            provider =eval(Engine_Configuration().ControlNet_provider)
        else:
            provider =Engine_Configuration().ControlNet_provider
        print(f"Loading ControlNET model:{ControlNET_drop},in:{provider}")       

        available_models=dict(ControlNet_config().available_controlnet_models())
 

        opts = ort.SessionOptions()
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
        opts.log_severity_level=3   
        self.controlnet_Model_ort= None

        self.ControlNET_Name=ControlNET_drop
        ControlNet_path = available_models[ControlNET_drop]
        self.controlnet_Model_ort = OnnxRuntimeModel.from_pretrained(ControlNet_path, sess_options=opts, provider=provider)
      
        return self.controlnet_Model_ort

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
        opts.log_severity_level=3
        print(f"Scheduler:{sched_name}")        
        sched_pipe=SchedulersConfig().scheduler(sched_name,model_path)
        if self.ControlNet_pipe == None:
            print(f"Using modified model for ControlNET:{model_path}")            
            self.controlnet_Model_ort= self.__load_ControlNet_model(model_path,ControlNET_drop)
            #self.controlnet_unet_ort= self.__load_uNet_model(model_path)

            self.ControlNet_pipe = OnnxStableDiffusionControlNetPipeline.from_pretrained(
                #unet=self.controlnet_unet_ort,
                model_path,
                controlnet=self.controlnet_Model_ort,
                vae_encoder=Vae_and_Text_Encoders().vae_encoder,
                vae_decoder=Vae_and_Text_Encoders().vae_decoder,
                text_encoder=Vae_and_Text_Encoders().text_encoder,
                scheduler=sched_pipe,
                sess_options=opts, 
                provider = Engine_Configuration().MAINPipe_provider,
                requires_safety_checker= False
            )
        else:
            self.ControlNet_pipe.scheduler= sched_pipe
            if self.ControlNET_Name!=ControlNET_drop:
                self.ControlNet_pipe.controlnet= None
                gc.collect()
                self.__load_ControlNet_model(model_path,ControlNET_drop)
                self.ControlNet_pipe.controlnet= self.controlnet_Model_ort

        return self.ControlNet_pipe

    def run_inference(self,prompt,neg_prompt,input_image,width,height,eta,steps,guid,seed,pose_image=None,controlnet_conditioning_scale=1.0):
        import numpy as np
        rng = np.random.RandomState(int(seed))
        image = self.ControlNet_pipe(
            prompt,
            input_image,
            negative_prompt=neg_prompt,
            width = width,
            height = height,
            num_inference_steps = steps,
            guidance_scale=guid,
            eta=eta,
            num_images_per_prompt=1,
            generator=rng,
            controlnet_conditioning_scale=controlnet_conditioning_scale
        ).images[0]
        #Añadir el diccionario
        dictio={'prompt':prompt,'neg_prompt':neg_prompt,'steps':steps,'guid':guid,'eta':eta,'strength':controlnet_conditioning_scale,'seed':seed}        
        return image,dictio

    def create_seeds(self,seed=None,iter=1,same_seeds=False):
        self.seeds=seed_generator(seed,iter)
        if same_seeds:
            for seed in self.seeds:
                seed = self.seeds[0]

    def unload_from_memory(self):
        self.ControlNet_pipe= None
        self.controlnet_Model_ort= None
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