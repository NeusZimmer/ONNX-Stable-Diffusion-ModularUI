import gradio as gr
import os,gc,re
from Engine.General_parameters import Engine_Configuration as Engine_config
from Engine.General_parameters import running_config
from Engine.General_parameters import UI_Configuration
from Engine import pipelines_engines
from UI import UI_common_funcs as UI_common
from Engine import engine_common_funcs as Engine_common
from PIL import Image, PngImagePlugin

global next_prompt
global processed_images
processed_images=[]
next_prompt=None
global number_of_passes


def show_HiRes_txt2img_ui():
    model_list = UI_common.get_model_list("txt2img")
    sched_list = get_schedulers_list()
    ui_config=UI_Configuration()
    gr.Markdown("Start typing below and then click **Generate** to see the output.")
    with gr.Row(): 
        with gr.Column(scale=13, min_width=650):
            model_drop = gr.Dropdown(model_list, value=(model_list[0] if len(model_list) > 0 else None), label="model folder", interactive=True)
            with gr.Accordion("Partial Reloads",open=False):
                reload_vae_btn = gr.Button("VAE Decoder:Apply Changes & Reload")
                #reload_model_btn = gr.Button("Model:Apply new model & Fast Reload Pipe")

            sch_t0 = gr.Radio(sched_list, value=sched_list[0], label="scheduler")
            prompt_t0 = gr.Textbox(value="", lines=2, label="prompt")
            neg_prompt_t0 = gr.Textbox(value="", lines=2, label="negative prompt")

            with gr.Accordion(label="Latents experimentals",open=False):
                #multiplier = gr.Slider(0, 1, value=0.18215, step=0.05, label="Multiplier, blurry the ingested latent, 1 to do not modify", interactive=True)
                #strengh_t0 = gr.Slider(0, 1, value=0.8, step=0.05, label="Strengh, or % of steps to apply the latent", interactive=True)
                #offset_t0 = gr.Slider(0, 100, value=1, step=1, label="Offset Steps for the scheduler", interactive=True)
                #latents_experimental1 = gr.Checkbox(label="Save generated latents ", value=False, interactive=True)
                latents_experimental2 = gr.Checkbox(label="Load latent from a generation", value=False, interactive=True)
                name_of_latent = gr.Textbox(value="", lines=1, label="Names of Numpy File Latents (1:x.npy,2:y.pt,3:noise-(width)xheight())")
                latent_formula = gr.Textbox(value="", lines=1, label="Formula for the sumatory of latents")


            with gr.Row():
                gr.Markdown("Common parameters")

            with gr.Row():
                iter_t0 = gr.Slider(1, 100, value=1, step=1, label="iteration count")
                guid_t0 = gr.Slider(0, 50, value=7.5, step=0.1, label="guidance")
                batch_t0 = gr.Slider(1, 4, value=1, step=1, label="batch size", interactive=False,visible=False)
            with gr.Row():                
                eta_t0 = gr.Slider(0, 1, value=0.0, step=0.01, label="DDIM eta", interactive=True)
                seed_t0 = gr.Textbox(value="", max_lines=1, label="seed")
                fmt_t0 = gr.Radio(["png", "jpg"], value="png", label="image format", interactive=False,visible=False)
            
            with gr.Row():
                gr.Markdown("HiRes parameters")
            with gr.Row():                         
                height_t1 = gr.Slider(64, 2048, value=512, step=64, label="HiRes height")
                width_t1 = gr.Slider(64, 2048, value=512, step=64, label="HiRes width")
            with gr.Row():                
                steps_t1 = gr.Slider(1, 100, value=16, step=1, label="HiRes steps")
                hires_passes_t1 = gr.Slider(1, 10, value=2, step=1, label="HiRes passes")

            with gr.Row():
                gr.Markdown("First pass parameters")
            with gr.Row():                
                height_t0 = gr.Slider(64, 2048, value=512, step=64, label="height")
                width_t0 = gr.Slider(64, 2048, value=512, step=64, label="width")
                steps_t0 = gr.Slider(1, 3000, value=16, step=1, label="steps")                

            with gr.Row():
                gr.Markdown("Other parameters")
                save_textfile=gr.Checkbox(label="Save prompt into a txt file")
                save_low_res=gr.Checkbox(label="Save generated Low-Res Img")

        with gr.Column(scale=11, min_width=550):
            with gr.Row():
                gen_btn = gr.Button("Generate", variant="primary", elem_id="gen_button")
                clear_btn = gr.Button("Cancel",info="Cancel at end of current iteration",variant="stop", elem_id="gen_button")
                memory_btn = gr.Button("Release memory", elem_id="mem_button")

            if ui_config.wildcards_activated:
                from UI import styles_ui
                styles_ui.show_styles_ui()
                with gr.Accordion(label="Live Prompt & Wildcards for multiple iterations",open=False):
                    with gr.Row():
                        next_wildcard = gr.Textbox(value="",lines=4, label="Next Prompt", interactive=True)
                        discard = gr.Textbox(value="", label="Discard", visible=False, interactive=False)
                    with gr.Row():
                        wildcard_show_btn = gr.Button("Show next prompt", elem_id="wildcard_button")
                        wildcard_gen_btn = gr.Button("Regenerate next prompt", variant="primary", elem_id="wildcard_button")
                        wildcard_apply_btn = gr.Button("Use edited prompt", elem_id="wildcard_button")
            with gr.Row():
                image_out = gr.Gallery(value=None, label="output images")
            with gr.Accordion(label="Low Res output images",open=False):
                with gr.Row():
                    low_res_image_out = gr.Gallery(value=None, label="Low res output images")
            with gr.Row():
                status_out = gr.Textbox(value="", label="status")
            with gr.Row():
                Selected_image_status= gr.Textbox(value="", label="status",visible=True)
                Selected_image_index= gr.Number(show_label=False, visible=False)

  
    image_out.select(fn=get_select_index, inputs=[image_out,status_out], outputs=[Selected_image_index,Selected_image_status])
    clear_btn.click(fn=UI_common.cancel_iteration,inputs=None,outputs=None)
    reload_vae_btn.click(fn=change_vae,inputs=model_drop,outputs=None)
    #reload_model_btn.click(fn=change_model,inputs=model_drop,outputs=None)

    list_of_All_Parameters2=[model_drop,prompt_t0,neg_prompt_t0,sch_t0,iter_t0,batch_t0,steps_t0,steps_t1,guid_t0,height_t0,width_t0,height_t1,width_t1,eta_t0,seed_t0,fmt_t0,hires_passes_t1,save_textfile, save_low_res,latent_formula,name_of_latent,latents_experimental2]    

    memory_btn.click(fn=UI_common.clean_memory_click, inputs=None, outputs=None)    
    

    gen_btn.click(fn=generate_click,inputs=list_of_All_Parameters2,outputs=[image_out,status_out,low_res_image_out])

    if ui_config.wildcards_activated:
        wildcard_gen_btn.click(fn=gen_next_prompt, inputs=prompt_t0, outputs=[discard,next_wildcard])
        wildcard_show_btn.click(fn=show_next_prompt, inputs=None, outputs=next_wildcard)
        wildcard_apply_btn.click(fn=apply_prompt, inputs=next_wildcard, outputs=None)



def change_vae(model_drop):
    from Engine.pipelines_engines import txt2img_pipe
    from Engine.pipelines_engines import Vae_and_Text_Encoders
    pipe=txt2img_pipe().txt2img_pipe
    vae=Vae_and_Text_Encoders()
    pipe.vae_decoder=vae.load_vaedecoder(f"{UI_Configuration().models_dir}\{model_drop}")
    return


def get_select_index(image_out,status_out, evt:gr.SelectData):
    status_out=eval(status_out)
    global number_of_passes
    number_of_passes
    resto=evt.index % number_of_passes
    index= (evt.index-resto)/number_of_passes

    return index,status_out[int(index)]
    #return evt.index,status_out[int(index)]

def gallery_view(images,dict_statuses):
    return images[0]


def get_schedulers_list():
    sched_config = pipelines_engines.SchedulersConfig()
    sched_list =sched_config.available_schedulers
    return sched_list

def select_scheduler(sched_name,model_path):
    return pipelines_engines.SchedulersConfig().scheduler(sched_name,model_path)

def gen_next_prompt(prompt_t0,initial=False):
    global next_prompt
    Running_information= running_config().Running_information    
    style=Running_information["Style"]

    style_pre =""
    style_post=""
    if style:
        styles=style.split("|")
        style_pre =styles[0]
        style_post=styles[1]

    if (initial):
        next_prompt=None
        prompt=prompt_t0
    else:
        if (next_prompt != None):
            prompt=next_prompt
        else:
            prompt=wildcards_process(prompt_t0)

        next_prompt=wildcards_process(prompt_t0)
    prompt = style_pre+" " +prompt+" " +style_post
    #print(f"Prompt:{prompt}")
    return prompt,next_prompt

def apply_prompt(prompt):
    global next_prompt
    next_prompt=prompt

def wildcards_process(prompt):
    from Scripts import wildcards
    wildcard=wildcards.WildcardsScript()
    new_prompt,discarded=wildcard.process(prompt)
    return new_prompt

def show_next_prompt():
    global next_prompt
    return next_prompt


def generate_click(
    model_drop,prompt_t0,neg_prompt_t0,sch_t0,
    iter_t0,batch_t0,steps_t0,steps_t1,guid_t0,height_t0,
    width_t0,height_t1,width_t1,eta_t0,seed_t0,fmt_t0,
    hires_passes_t1,save_textfile, save_low_res,
    latent_formula,name_of_latent,latents_experimental2):

    from Engine.General_parameters import running_config

    if latents_experimental2:
        running_config().Running_information.update({"Load_Latents":True})
        running_config().Running_information.update({"Latent_Name":name_of_latent})
        running_config().Running_information.update({"Latent_Formula":latent_formula})
    else:
        running_config().Running_information.update({"Load_Latents":False})

    from Engine.Pipelines.txt2img_hires import txt2img_hires_pipe

    global number_of_passes
    number_of_passes=hires_passes_t1

    Running_information= running_config().Running_information
    Running_information.update({"Running":True})

    if (Running_information["model"] != model_drop or Running_information["tab"] != "hires_txt2img"):
        UI_common.clean_memory_click()
        Running_information.update({"model":model_drop})
        Running_information.update({"tab":"hires_txt2img"})

    model_path=UI_Configuration().models_dir+"\\"+model_drop
    txt2img_hires_pipe().initialize(model_path,sch_t0)
    txt2img_hires_pipe().create_seeds(seed_t0,iter_t0,False)
    images= []
    images_low= []    
    information=[]
    counter=1
    img_index=Engine_common.get_next_save_index(output_path=UI_Configuration().output_path)

    for seed in txt2img_hires_pipe().seeds:
        if running_config().Running_information["cancelled"]:
            break
        prompt,discard=gen_next_prompt(prompt_t0)
        print(f"Iteration:{counter}/{iter_t0}")
        counter+=1
        #batch_images,info = txt2img_hires_pipe().run_inference(
        lowres_image,hires_images,info = txt2img_hires_pipe().run_inference(
            prompt,
            neg_prompt_t0,
            hires_passes_t1,
            height_t0,
            width_t0,
            height_t1,
            width_t1,
            steps_t0,
            steps_t1,
            guid_t0,
            eta_t0,
            batch_t0,
            seed)
        for hires_image in hires_images:
            images.append(hires_image)
        images_low.append(lowres_image)
        info=dict(info)
        info['Sched:']=sch_t0
        information.append(info)

        style= running_config().Running_information["Style"]
        Engine_common.save_image(hires_images,info,img_index,UI_Configuration().output_path,style,save_textfile)
        img_index+=1
        if save_low_res:
            Engine_common.save_image([lowres_image],info,img_index,UI_Configuration().output_path)
            img_index+=1          
    
    running_config().Running_information.update({"cancelled":False})
    gen_next_prompt("",True)
    Running_information.update({"Running":False})
    #return images,information

    return images,information,images_low