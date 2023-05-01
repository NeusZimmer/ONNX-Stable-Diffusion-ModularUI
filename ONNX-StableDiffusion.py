import gradio as gr
from Engine.General_parameters import Engine_Configuration
from Engine.General_parameters import UI_Configuration
from Engine.General_parameters import running_config
global demo

def init_ui():
    ui_config=UI_Configuration()
    with gr.Blocks(title="ONNX Difussers Modular UI") as demo:
        if ui_config.Txt2img_Tab:
            with gr.Tab(label="Txt2img Pipelines & Inferences") as tab0:
                from UI import txt2img_ui as txt2img_ui
                txt2img_ui.show_txt2img_ui()
        if ui_config.Img2Img_Tab:
            with gr.Tab(label="Img2Img") as tab1:
                from UI import Img2Img_ui
                Img2Img_ui.show_Img2Img_ui()
        if ui_config.InPaint_Tab:
            with gr.Tab(label="InPaint") as tab2:
                from UI import Inpaint_ui
                Inpaint_ui.show_Inpaint_ui()
        if ui_config.Tools_Tab:
            with gr.Tab(label="Image Tools") as tab3:
                from UI import ui_image_tools
                ui_image_tools.show_danbooru_area()
                ui_image_tools.show_image_resolution_area()
        if ui_config.InstructP2P_Tab:
            with gr.Tab(label="Instruct Pix2Pix") as tab4:
                from UI import instructp2p_ui
                instructp2p_ui.show_instructp2p_ui()
        with gr.Tab(label="Configuration") as tab5:
            from UI import config_ui_general
            config_ui_general.show_general_configuration()
            if ui_config.Advanced_Config:
                    from UI import config_ui_engine as config_ui_engine
                    config_ui_engine.show_providers_configuration()
                    #from UI import config_ui_wildcards as wilcards_ui_config
                    #wilcards_ui_config.show_wilcards_configuration()
        #with gr.Box():
            #restart_btn = gr.Button("Restart UI", variant="primary")
            #restart_btn.click(fn=restart, inputs=None, outputs=None)
    return demo

def restart():
    import gc
    global demo
    demo.close()
    gr.close_all()
    demo = init_ui()
    print("Restart?")
    gc.collect()
    demo.launch(server_port=UI_Configuration().GradioPort)




Running_information= running_config().Running_information
Running_information.update({"cancelled":False})
Running_information.update({"model":""})
Running_information.update({"tab":""})
Running_information.update({"Running":False})

Engine_Configuration().load_config_json()
demo =init_ui()
demo.launch(server_port=UI_Configuration().GradioPort)


