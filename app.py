# app.py
import gradio as gr
import torch
from diffusers import StableDiffusionPanoramaPipeline, DPMSolverMultistepScheduler
from PIL import Image

# Load model
pipe = StableDiffusionPanoramaPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-base",
    torch_dtype=torch.float16
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

def generate_room_panorama(prompt, style, num_steps=30):
    full_prompt = f"{prompt}, {style} style, cylindrical panorama, 360 degree view, high quality interior design"
    
    image = pipe(
        full_prompt,
        height=512,
        width=2048,  # Wider for panorama effect
        num_inference_steps=num_steps,
        guidance_scale=7.5
    ).images[0]
    
    return image

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Room Panorama Generator")
    
    with gr.Row():
        with gr.Column():
            room_type = gr.Dropdown(
                ["living room", "bedroom", "kitchen", "bathroom", "office"],
                label="Room Type"
            )
            style = gr.Dropdown(
                ["modern", "minimalist", "industrial", "scandinavian", "traditional"],
                label="Design Style"
            )
            custom_prompt = gr.Textbox(label="Additional Details", placeholder="e.g., large windows, wooden floor")
            steps = gr.Slider(20, 50, value=30, step=5, label="Quality (steps)")
            generate_btn = gr.Button("Generate Panorama")
        
        with gr.Column():
            output_image = gr.Image(label="Generated Panorama")
    
    def create_panorama(room, style, custom, steps):
        prompt = f"{room} interior, {custom}"
        return generate_room_panorama(prompt, style, steps)
    
    generate_btn.click(
        create_panorama,
        inputs=[room_type, style, custom_prompt, steps],
        outputs=output_image
    )

demo.launch()