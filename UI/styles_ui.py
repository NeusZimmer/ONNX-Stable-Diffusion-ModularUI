import gradio as gr
#from Engine.General_parameters import ControlNet_config
from UI import styles_ui
global styles_dict

def show_styles_ui():
    global styles_dict
    styles_dict= get_styles()
    styles_keys= list(styles_dict.keys())
    if True:
        with gr.Accordion(label="Styles",open=False):
            gr.Markdown("Use your preferred Styles")
            with gr.Row():
                with gr.Column(scale=1):
                    Style_Select = gr.Radio(styles_keys,value=styles_keys[0],label="Available Styles")
                with gr.Column(scale=8):
                    styletext_pre = gr.Textbox(value="", lines=2, label="Style previous text")
                    styletext_post = gr.Textbox(value="", lines=2, label="Style posterior text")
            with gr.Row():
                #apply_btn = gr.Button("Apply Styles")
                save_btn = gr.Button("Apply & Save Styles")
                save_check = gr.Checkbox(label="Save in memory styles to disk", value=False, interactive=True)                


        all_inputs=[Style_Select,styletext_pre,styletext_post,save_check]

        save_btn.click(fn=save_styles, inputs=all_inputs, outputs=None)
        Style_Select.change(fn=apply_styles, inputs=Style_Select, outputs=[styletext_pre,styletext_post])

def get_styles():
    import json
    """dict={
            "None":True,
            "StudioPhoto":"(RAW, 8k) |, studio lights,pseudo-impasto",
            "Style1":"(cartoon) |, Ink drawing line art",
            "Style2":"unity wallpaper, 8k, high quality, | masterpiece,(masterpiece, top quality, best quality)"
        }"""
    with open('./Engine/config_files/Styles.json', 'r') as openfile:
        jsonStr = json.load(openfile)

    #print(jsonStr)
    #print(type(jsonStr))
    jsonStr.update({"None":True})
    return jsonStr


def apply_styles(*args):
    global styles_dict
    dict=styles_dict
    style=args[0]

    from Engine.General_parameters import running_config
    Running_information= running_config().Running_information    
    Running_information.update({"Style":False})

    if style != "None":
        print(dict[style])
        Running_information.update({"Style":dict[style]})
        params=dict[style].split("|")
        return params[0],params[1]        
    else:
        return "",""

def save_styles(*args):
    import json
    global styles_dict
    styles_dict

    style=args[0]
    style_pre=args[1]
    style_post=args[2]
    style_save=args[3]            

    from Engine.General_parameters import running_config
    Running_information= running_config().Running_information    
    Running_information.update({"Style":False})

    if style != "None":
        styles_dict.update({style:f"{style_pre}|{style_post}"})
        Running_information.update({"Style":styles_dict[style]})

    if style_save:
        jsonStr = json.dumps(styles_dict)
        with open("./Engine/config_files/Styles.json", "w") as outfile:
            outfile.write(jsonStr)
        print("Saving Styles")


