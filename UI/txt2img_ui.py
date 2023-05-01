import gradio as gr
import os,gc,re
from Engine.General_parameters import Engine_Configuration as Engine_config
from Engine.General_parameters import running_config
from Engine.General_parameters import UI_Configuration
from Engine import pipelines_engines
from PIL import Image, PngImagePlugin

global next_prompt
global processed_images
processed_images=[]
next_prompt=None


def show_txt2img_ui():
    model_list = get_model_list()
    sched_list = get_schedulers_list()
    ui_config=UI_Configuration()
    gr.Markdown("Start typing below and then click **Generate** to see the output.")
    with gr.Row(): 
        with gr.Column(scale=13, min_width=650):
            model_drop = gr.Dropdown(model_list, value=(model_list[0] if len(model_list) > 0 else None), label="model folder", interactive=True)
            with gr.Accordion(label="Use Alternative VAE",open=False):
                forced_vae = gr.Checkbox(label="Activate", value=False, interactive=True)
                path_to_vae = gr.Textbox(value=ui_config.forced_VAE_Dir, lines=1, label="Alternative VAE")
            prompt_t0 = gr.Textbox(value="", lines=2, label="prompt")
            neg_prompt_t0 = gr.Textbox(value="", lines=2, label="negative prompt")
            sch_t0 = gr.Radio(sched_list, value=sched_list[0], label="scheduler")
            with gr.Row():
                iter_t0 = gr.Slider(1, 100, value=1, step=1, label="iteration count")
                batch_t0 = gr.Slider(1, 4, value=1, step=1, label="batch size")
            steps_t0 = gr.Slider(1, 300, value=16, step=1, label="steps")
            guid_t0 = gr.Slider(0, 50, value=7.5, step=0.1, label="guidance")
            height_t0 = gr.Slider(256, 2048, value=512, step=64, label="height")
            width_t0 = gr.Slider(256, 2048, value=512, step=64, label="width")
            eta_t0 = gr.Slider(0, 1, value=0.0, step=0.01, label="DDIM eta", interactive=False)
            seed_t0 = gr.Textbox(value="", max_lines=1, label="seed")
            fmt_t0 = gr.Radio(["png", "jpg"], value="png", label="image format")
        with gr.Column(scale=11, min_width=550):
            with gr.Row():
                gen_btn = gr.Button("Generate", variant="primary", elem_id="gen_button")
                clear_btn = gr.Button("Cancel",info="Cancel at end of current iteration",variant="stop", elem_id="gen_button")
                memory_btn = gr.Button("Release memory", elem_id="mem_button")
                #test_btn = gr.Button("Test")
            if ui_config.wildcards_activated:
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

            with gr.Row():
                status_out = gr.Textbox(value="", label="status")
            with gr.Row():
                Selected_image_status= gr.Textbox(value="", label="status",visible=True)
                Selected_image_index= gr.Number(show_label=False, visible=False)
            with gr.Accordion(label="Edit/Save gallery images",open=False):
                with gr.Row():
                    delete_btn = gr.Button("Delete", elem_id="gallery_button")
                    save_btn = gr.Button("Save", elem_id="gallery_button")
                    #previous_btn = gr.Button("Previous", elem_id="gallery_button")
                    #next_btn = gr.Button("Next", elem_id="gallery_button")
  
    image_out.select(fn=get_select_index, inputs=[image_out,status_out], outputs=[Selected_image_index,Selected_image_status])
    delete_btn.click(fn=delete_selected_index, inputs=[Selected_image_index,status_out], outputs=[image_out,status_out])
    clear_btn.click(fn=cancel_iteration,inputs=None,outputs=None)
    forced_vae.change(fn=change_vae,inputs=[forced_vae,path_to_vae],outputs=None)

    list_of_All_Parameters=[model_drop,prompt_t0,neg_prompt_t0,sch_t0,iter_t0,batch_t0,steps_t0,guid_t0,height_t0,width_t0,eta_t0,seed_t0,fmt_t0]
    gen_btn.click(fn=generate_click, inputs=list_of_All_Parameters, outputs=[image_out,status_out])
    #sch_t0.change(fn=select_scheduler, inputs=sch_t0, outputs= None)  #Atencion cambiar el DDIM ETA si este se activa
    memory_btn.click(fn=clean_memory_click, inputs=None, outputs=None)
    #test_btn.click(fn=test1,inputs=[model_drop,prompt_t0,neg_prompt_t0,sch_t0],outputs=image_out)

    if ui_config.wildcards_activated:
        wildcard_gen_btn.click(fn=gen_next_prompt, inputs=prompt_t0, outputs=[discard,next_wildcard])
        wildcard_show_btn.click(fn=show_next_prompt, inputs=None, outputs=next_wildcard)
        wildcard_apply_btn.click(fn=apply_prompt, inputs=next_wildcard, outputs=None)


def change_vae(forced_vae,path_to_vae):
    from Engine.shared_params import UI_Configuration as UI_Configuration
    ui_config=UI_Configuration()
    ui_config.Forced_VAE =forced_vae
    if ui_config.Forced_VAE:
        ui_config.forced_VAE_Dir =path_to_vae
    #else:
    #    ui_config.forced_VAE_Dir =""
    return


def delete_selected_index(Selected_image_index,status_out):
    global processed_images
    Selected_image_index=int(Selected_image_index)
    status_out=eval(status_out)
    processed_images.pop(Selected_image_index)
    status_out.pop(Selected_image_index)
    return processed_images,status_out


def get_select_index(image_out,status_out, evt:gr.SelectData):
    status_out=eval(status_out)
    return evt.index,status_out[evt.index]

def gallery_view(images,dict_statuses):
    print(dict_statuses)
    return images[0]

def get_model_list():
    model_list = []
    try:
        with os.scandir(UI_Configuration().models_dir) as scan_it:
            for entry in scan_it:
                if entry.is_dir():
                    model_list.append(entry.name)
    except:
        model_list.append("Models directory does not exist, configure it")
    return model_list

def get_schedulers_list():
    sched_config = pipelines_engines.SchedulersConfig()
    sched_list =sched_config.available_schedulers
    return sched_list

def select_scheduler(sched_name,model_path):
    return pipelines_engines.SchedulersConfig().scheduler(sched_name,model_path)

def gen_next_prompt(prompt_t0,initial=False):
    global next_prompt
    if (initial):
        next_prompt=None
        prompt=prompt_t0
    else:
        if (next_prompt != None):
            prompt=next_prompt
        else:
            prompt=wildcards_process(prompt_t0)

        next_prompt=wildcards_process(prompt_t0)
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

def generate_click(model_drop,prompt_t0,neg_prompt_t0,sch_t0,iter_t0,batch_t0,steps_t0,guid_t0,height_t0,width_t0,eta_t0,seed_t0,fmt_t0):
    from Engine.pipelines_engines import txt2img_pipe

    Running_information= running_config().Running_information
    Running_information.update({"Running":True})


    if (Running_information["model"] != model_drop or Running_information["tab"] != "txt2img"):
        clean_memory_click()
        Running_information.update({"model":model_drop})
        Running_information.update({"tab":"txt2img"})

    model_path=UI_Configuration().models_dir+"\\"+model_drop
    pipe=txt2img_pipe().initialize(model_path,sch_t0)

    txt2img_pipe().create_seeds(seed_t0,iter_t0,False)
    images= []
    information=[]
    counter=1
    img_index=get_next_save_index()
    for seed in txt2img_pipe().seeds:
        if running_config().Running_information["cancelled"]:
            running_config().Running_information.update({"cancelled":False})
            break
        prompt,discard=gen_next_prompt(prompt_t0)
        running_config().parse_prompt_attention(prompt)
        print(f"Iteration:{counter}/{iter_t0}")
        counter+=1
        #print("resultado de promptattention")
        batch_images,info = txt2img_pipe().run_inference(
            prompt,
            neg_prompt_t0,
            height_t0,
            width_t0,
            steps_t0,
            guid_t0,
            eta_t0,
            batch_t0,
            seed)
        images.extend(batch_images)
        info=dict(info)
        info['Sched:']=sch_t0
        information.append(info)
        save_image(batch_images,info,img_index)
        img_index+=1

    global processed_images
    processed_images=images
    gen_next_prompt("",True)
    Running_information.update({"Running":False})
    return images,information


def get_next_save_index():
    output_path=UI_Configuration().output_path
    dir_list = os.listdir(output_path)
    if len(dir_list):
        pattern = re.compile(r"([0-9][0-9][0-9][0-9][0-9][0-9])-([0-9][0-9])\..*")
        match_list = [pattern.match(f) for f in dir_list]
        next_index = max([int(m[1]) if m else -1 for m in match_list]) + 1
    else:
        next_index = 0
    return next_index


def save_image(batch_images,info,next_index):
    output_path=UI_Configuration().output_path

    info_png = f"{info}"
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("parameters",info_png)
    prompt=info["prompt"]
    short_prompt = prompt.strip("<>:\"/\\|?*\n\t")
    short_prompt = re.sub(r'[\\/*?:"<>|\n\t]', "", short_prompt)
    short_prompt = short_prompt[:49] if len(short_prompt) > 50 else short_prompt

    os.makedirs(output_path, exist_ok=True)
    """dir_list = os.listdir(output_path)
    if len(dir_list):
        pattern = re.compile(r"([0-9][0-9][0-9][0-9][0-9][0-9])-([0-9][0-9])\..*")
        match_list = [pattern.match(f) for f in dir_list]
        next_index = max([int(m[1]) if m else -1 for m in match_list]) + 1
    else:
        next_index = 0"""
    for image in batch_images:
        image.save(os.path.join(output_path,f"{next_index:06}-00.{short_prompt}.png",),optimize=True,pnginfo=metadata,)
    
def clean_memory_click():
    print("Cleaning memory")
    pipelines_engines.Vae_and_Text_Encoders().unload_from_memory()
    pipelines_engines.txt2img_pipe().unload_from_memory()
    pipelines_engines.inpaint_pipe().unload_from_memory()
    pipelines_engines.instruct_p2p_pipe().unload_from_memory()
    pipelines_engines.img2img_pipe().unload_from_memory()
    gc.collect()

def cancel_iteration():
    running_config().Running_information.update({"cancelled":True})
    print("\nCancelling at the end of the current iteration")
    
