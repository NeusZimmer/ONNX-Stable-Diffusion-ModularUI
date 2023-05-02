
# Singleton/BorgSingleton.py
# Alex Martelli's 'Borg'
# https://python-3-patterns-idioms-test.readthedocs.io/en/latest/Singleton.html
import json

class Borg:
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

class UI_Configuration(Borg):
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
    Tools_Tab = None
    Advanced_Config = None
    Forced_VAE = False
    GradioPort = 7860

    def __init__(self):
        Borg.__init__(self)
        if not self.__loaded:
            self.load_config()

    def __str__(self): return json.dumps(self.__dict__)

    def __load_default_values(self):
        import os
        self.models_dir=os.getcwd()+"\\models"
        self.output_path=os.getcwd()+"\\output"
        self.forced_VAE_Dir=os.getcwd()
        self.forced_ControlNet_dir=os.getcwd()
        self.Txt2img_Tab = True
        self.InPaint_Tab = True
        self.Img2Img_Tab = True
        self.Tools_Tab = True
        self.InstructP2P_Tab = True
        self.Advanced_Config = True
        self.Forced_VAE = False
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
                self.Tools_Tab = jsonStr["Tools_Tab"]
                self.Advanced_Config = jsonStr["Advanced_Config"]
                self.GradioPort = int(jsonStr["GradioPort"])
  
        except OSError:
            self.__load_default_values()
        return self

    def load_config(self):
        self.__load_config_json()
        self.__loaded=True


class running_config(Borg):
    Running_information= dict({"loaded":False})

    def __init__(self):
        Borg.__init__(self)
        if not self.Running_information["loaded"]==True:
            self.Running_information.update({"loaded":True})

    def __str__(self): return json.dumps(self.__dict__)

    def parse_prompt_attention(self,text):
        """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text

    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
        """
        import re
        re_attention = re.compile(r"""
        \\\(|
        \\\)|
        \\\[|
        \\]|
        \\\\|
        \\|
        \(|
        \[|
        :([+-]?[.\d]+)\)|
        \)|
        ]|
        [^\\()\[\]:]+|
        :
        """, re.X)
        re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)
        res = []
        round_brackets = []
        square_brackets = []

        round_bracket_multiplier = 1.1
        square_bracket_multiplier = 1 / 1.1

        def multiply_range(start_position, multiplier):
            for p in range(start_position, len(res)):
                res[p][1] *= multiplier

        for m in re_attention.finditer(text):
            text = m.group(0)
            weight = m.group(1)

            if text.startswith('\\'):
                res.append([text[1:], 1.0])
            elif text == '(':
                round_brackets.append(len(res))
            elif text == '[':
                square_brackets.append(len(res))
            elif weight is not None and len(round_brackets) > 0:
                multiply_range(round_brackets.pop(), float(weight))
            elif text == ')' and len(round_brackets) > 0:
                multiply_range(round_brackets.pop(), round_bracket_multiplier)
            elif text == ']' and len(square_brackets) > 0:
                multiply_range(square_brackets.pop(), square_bracket_multiplier)
            else:
                parts = re.split(re_break, text)
                for i, part in enumerate(parts):
                    if i > 0:
                        res.append(["BREAK", -1])
                    res.append([part, 1.0])

        for pos in round_brackets:
            multiply_range(pos, round_bracket_multiplier)

        for pos in square_brackets:
            multiply_range(pos, square_bracket_multiplier)

        if len(res) == 0:
            res = [["", 1.0]]

        # merge runs of identical weights
        i = 0
        while i + 1 < len(res):
            if res[i][1] == res[i + 1][1]:
                res[i][0] += res[i + 1][0]
                res.pop(i + 1)
            else:
                i += 1

        return res

