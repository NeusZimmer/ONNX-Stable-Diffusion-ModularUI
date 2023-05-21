
# Singleton/BorgSingleton.py
# Alex Martelli's 'Borg'
# https://python-3-patterns-idioms-test.readthedocs.io/en/latest/Singleton.html
import json

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


class Engine_Configuration(Borg):
    MAINPipe_provider="Not Selected"
    Scheduler_provider="Not Selected"
    ControlNet_provider="Not Selected"
    VAEDec_provider="Not Selected"
    TEXTEnc_provider="Not Selected"
    DeepDanBooru_provider="Not Selected"

    #MAINPipe_provider,Scheduler_provider,ControlNet_provider,VAEDec_provider,TEXTEnc_provider
    def __init__(self):
        Borg.__init__(self)

    def __str__(self): return json.dumps(self.__dict__)

    def save_config_json(self):
        jsonStr = json.dumps(self.__dict__)
        print(type(jsonStr))
        with open("EngineConfig.json", "w") as outfile:
            outfile.write(jsonStr)
        print(jsonStr)
        return jsonStr

    def load_default_values(self):
        print("Loading default provider values:CPU")
        self.MAINPipe_provider="CPUExecutionProvider"
        self.Scheduler_provider="CPUExecutionProvider"
        self.ControlNet_provider="CPUExecutionProvider"
        self.VAEDec_provider="CPUExecutionProvider"
        self.TEXTEnc_provider="CPUExecutionProvider"
        self.DeepDanBooru_provider="CPUExecutionProvider"
    
    def load_config_json(self):
        try:
            with open('EngineConfig.json', 'r') as openfile:
                jsonStr = json.load(openfile)
                self.MAINPipe_provider = jsonStr["MAINPipe_provider"]
                self.Scheduler_provider = jsonStr["Scheduler_provider"]
                self.ControlNet_provider = jsonStr["ControlNet_provider"]
                self.VAEDec_provider = jsonStr["VAEDec_provider"]
                self.TEXTEnc_provider = jsonStr["TEXTEnc_provider"]
                self.DeepDanBooru_provider = jsonStr["DeepDanBooru_provider"]
        except OSError:
            self.load_default_values()
        return self

class UI_Configuration(Borg1):
    __loaded= False
    models_dir=""
    output_path = ""
    wildcards_activated=True
    forced_VAE_Dir = None
    forced_ControlNet_dir =None
    Txt2img_Tab = None
    InPaint_Tab = None
    Img2Img_Tab = None
    InstructP2P_Tab = None
    ControlNet_Tab = None
    Tools_Tab = None
    Advanced_Config = None
    Forced_VAE = False
    Forced_ControlNet = False
    GradioPort = 7860

    def __init__(self):
        Borg1.__init__(self)
        if not self.__loaded:
            self.load_config()

    def __str__(self): return json.dumps(self.__dict__)

    def __load_default_values(self):
        import os
        self.models_dir=os.getcwd()+"\\models"
        self.output_path=os.getcwd()+"\\output"
        self.forced_VAE_Dir=os.getcwd()
        self.forced_ControlNet_dir=os.getcwd()+"\\models"
        self.Txt2img_Tab = True
        self.InPaint_Tab = True
        self.Img2Img_Tab = True
        self.Tools_Tab = True
        self.InstructP2P_Tab = True
        self.ControlNet_Tab = True
        self.Advanced_Config = True
        self.Forced_VAE = False
        self.Forced_ControlNet = False
        self.GradioPort = 7860

    def save_config_json(self):
        jsonStr = json.dumps(self.__dict__)
        with open("UIConfig.json", "w") as outfile:
            outfile.write(jsonStr)
        print(jsonStr)
        return jsonStr

    def __load_config_json(self):
        try:
            with open('UIConfig.json', 'r') as openfile:
                jsonStr = json.load(openfile)
                self.models_dir = jsonStr["models_dir"]
                self.output_path= jsonStr["output_path"]
                self.forced_VAE_Dir= jsonStr["forced_VAE_Dir"]
                self.forced_ControlNet_dir= jsonStr["forced_ControlNet_dir"]
                self.Txt2img_Tab = jsonStr["Txt2img_Tab"]
                self.InPaint_Tab = jsonStr["InPaint_Tab"]
                self.Img2Img_Tab = jsonStr["Img2Img_Tab"]
                self.InstructP2P_Tab = jsonStr["InstructP2P_Tab"]
                self.ControlNet_Tab = int(jsonStr["ControlNet_Tab"])
                self.Tools_Tab = jsonStr["Tools_Tab"]
                self.Advanced_Config = jsonStr["Advanced_Config"]
                self.GradioPort = int(jsonStr["GradioPort"])

        except OSError:
            self.__load_default_values()
        return self

    def load_config(self):
        self.__load_config_json()
        self.__loaded=True


class running_config(Borg2):
    Running_information= dict({"loaded":False})

    def __init__(self):
        Borg2.__init__(self)
        if not self.Running_information["loaded"]==True:
            self.Running_information.update({"loaded":True})

    def __str__(self): return json.dumps(self.__dict__)


class ControlNet_config(Borg3):
    config = None
    def __init__(self):
        Borg3.__init__(self)
        if  self.config == None:
            self.config = self.__load_controlnet_config()

    def __str__(self): return json.dumps(self.__dict__)

    def __load_controlnet_config(self):
        import json
        standard_config = None
        try:
            with open('.\\Engine\\ControlNet.json', 'r') as openfile:
                jsonStr = json.load(openfile)
            standard_config = dict(jsonStr)
        except OSError:
            standard_config = {
                "canny_active":False,
                "canny_path":"",
                "depth_active":False,
                "depth_path":"",
                "hed_active":False,
                "hed_path":"",
                "mlsd_active":False,
                "mlsd_path":"",
                "normal_active":False,
                "normal_path":"",
                "openpose_active":False,
                "openpose_path":"",
                "seg_active":False,
                "seg_path":"",
            }
        return standard_config

    def load_config_from_disk(self):
        self.config = self.__load_controlnet_config()
        self.available_controlnet_models()

    def save_controlnet_config(self,controlnet_config):
        print("Salvando??")
        print(type(controlnet_config))

        import json
        json_data=jsonstr=json.dumps(controlnet_config)
        with open(".\\Engine\\ControlNet.json", "w") as outfile:
            outfile.write(json_data)

    def available_controlnet_models(self):
        available=[]
        for key, value in self.config.items():
            if "active" in key and value == True:
                model=key.split('_')[0]
                available.append((model,self.config[model+"_path"]))
        return available   #a list of tuples (model, path)


