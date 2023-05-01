# Wildcards 
#V2 Extended to provide info on status area
#An extension modificated version of a script from https://github.com/jtkelm2/stable-diffusion-webui-1/blob/main/scripts/wildcards.py
#Extracted from module wildcards for AUTOMATIC111
#Allows you to use `__name__` syntax in your prompt to get a random line from a file named `name.txt` in the wildcards directory.
# 1- Create a directory "Scripts" in OnnxDifussersUI, save this file inside as wildcards.py
# 2- Put wildcard files in a directory named "wildcards" inside "Scripts"
# 3- Include following lines in onnxUI.py .Do not forget to delete the  comment mark at the beginning of each line to make it work (#)

# In  line 107 (just before:if current_pipe == "txt2img":"

#        # Manage WildCards
#        from Scripts import wildcards 
#        WildCards_Activated= True
#        if WildCards_Activated:
#            old_prompt=prompt
#            wildcard=wildcards.WildcardsScript()
#            new_prompt,string_replaced=wildcard.process(prompt)
#            prompt=str(new_prompt)
#            new_prompts.append(string_replaced)

#  Add this line just before the line 121, before  "elif current_pipe == "img2img":"
#            prompt=old_prompt

# Optional: add this two lines between line 271 ")" and 272 (if current_pipe == "img2img":) to save the generated prompt info to the png file.
#        if new_prompt != prompt:
#            info_png = info_png + f" Wildcards generated change: {string_replaced}"
# Line 604 -- after "global original_steps"
#    global new_prompts
#    new_prompts=[]
#
# Line 1034 
#    status=status+"\nWildcards changed:\n"+str(new_prompts)
#


import os
import random
import sys

warned_about_files = {}
wildcard_dir = os.getcwd()+"\Scripts"
#print(wildcard_dir)


class WildcardsScript():
    def title(self):
        return "Simple wildcards class for OnnxDiffusersUI"

    def replace_wildcard(self, text, gen):
        if " " in text or len(text) == 0:
            return text,False

        replacement_file = os.path.join(wildcard_dir, "wildcards", f"{text}.txt")
        if os.path.exists(replacement_file):
            with open(replacement_file, encoding="utf8") as f:
                changed_text=gen.choice(f.read().splitlines())
                if "__" in changed_text:
                    changed_text, not_used = self.process(changed_text)
                return changed_text,True
        else:
            if replacement_file not in warned_about_files:
                print(f"File {replacement_file} not found for the __{text}__ wildcard.", file=sys.stderr)
                warned_about_files[replacement_file] = 1

        return text,False

    def process(self, original_prompt):
        string_replaced=""
        new_prompt=""
        gen = random.Random()
        text_divisions=original_prompt.split("__")

        for chunk in text_divisions:
            text,changed=self.replace_wildcard(chunk, gen)
            if changed:
                string_replaced=string_replaced+"Wildcard:"+chunk+"-->"+text+","
                new_prompt=new_prompt+text
            else:
                new_prompt=new_prompt+text        
        
        return  new_prompt, string_replaced
