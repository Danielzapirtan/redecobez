# app.py
import gradio as gr
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os

# Download model from a public source (no token needed)
def load_model():
    # Option 1: Use CompVis original SD (publicly available)
    model_id = "CompVis/stable-diffusion-v1-4"
    
    # This model is fully public and doesn't require token
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    
    return pipe

pipe = load_model()

# PRESET CUSTOM DETAILS - EDIT THIS ARRAY
PRESET_DETAILS = {
    "dormitor": "pat, mobilier alb (bibliotecă elegantă, dulap, birou, scaun, noptieră), confortabil",
    "dormitor 2": "pat, mobilier alb (bibliotecă elegantă, dulap, TV), confortabil",
    "living": "temă crem deschis, canapea în L, 2 fotolii, TV, bibliotecă elegantă, perdele",
    "bucătărie": "electrocasnice moderne, blat de marmură, insulă cu scaune de bar, lămpi suspendate"
}

def update_custom_details(room_type):
    """Update custom_details textbox based on selected room_type"""
    return PRESET_DETAILS.get(room_type, "")

def generate_room_panorama(room_type, style, custom_details, width_multiplier, num_steps):
    # Create panoramic prompt
    base_prompt = f"{room_type.replace(' 2', '')} interior design, {style} style"
    if custom_details:
        base_prompt += f", {custom_details}"
    
    full_prompt = f"{base_prompt}, panoramic view, wide angle, professional architecture photography, highly detailed, 8k"
    
    negative_prompt = "blurry, distorted, low quality, cropped, out of frame, duplicate"
    
    # Generate wider image for panoramic effect
    width = 512 * width_multiplier
    height = 512
    
    image = pipe(
        full_prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=num_steps,
        guidance_scale=7.5
    ).images[0]
    
    return image

# Gradio interface
with gr.Blocks(title="Generator Panorame Camere") as demo:
    gr.Markdown("""
    # 🏠 Generator Panorame Camere
    Generează imagini panoramice cilindrice ale designurilor personalizate de camere fără tokenuri API!
    """)
    
    with gr.Row():
        with gr.Column():
            room_type = gr.Dropdown(
                choices=[
                    "dormitor",
                    "dormitor 2",
                    "living",
                    "bucătărie",
                    "baie",
                    "birou",
                    "sufragerie",
                    "cinematograf acasă",
                    "sală sport"
                ],
                value="living",
                label="Tip Cameră"
            )
            
            style = gr.Dropdown(
                choices=[
                    "modern",
                    "minimalist",
                    "industrial",
                    "scandinav",
                    "tradițional",
                    "contemporan",
                    "rustic",
                    "art deco",
                    "modern mid-century",
                    "zen japonez"
                ],
                value="modern",
                label="Stil Design"
            )
            
            custom_details = gr.Textbox(
                label="Detalii Adiționale",
                placeholder="Completat automat pe baza tipului de cameră. Poți edita.",
                value=PRESET_DETAILS["living"],
                lines=3
            )
            
            width_multiplier = gr.Slider(
                minimum=2,
                maximum=4,
                value=2,
                step=1,
                label="Lățime Panoramă (multiplicator)",
                info="Mai mare = vedere panoramică mai largă"
            )
            
            num_steps = gr.Slider(
                minimum=20,
                maximum=50,
                value=30,
                step=5,
                label="Pași Generare",
                info="Mai multe = calitate mai bună dar mai lent"
            )
            
            generate_btn = gr.Button("🎨 Generează Panorama", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(
                label="Panorama Generată",
                type="pil"
            )
            
            gr.Markdown("""
            ### Sfaturi:
            - Multiplicatorul de lățime 3-4 funcționează cel mai bine pentru vederi panoramice
            - Presetările de camere completează automat detaliile (editabile)
            - Adaugă detalii specifice pentru rezultate mai bune
            - Generarea durează ~30-60 secunde pe GPU Colab
            """)
    
    # Auto-update custom_details when room_type changes
    room_type.change(
        fn=update_custom_details,
        inputs=[room_type],
        outputs=[custom_details]
    )
    
    generate_btn.click(
        fn=generate_room_panorama,
        inputs=[room_type, style, custom_details, width_multiplier, num_steps],
        outputs=output_image
    )
    
    # Examples
    gr.Examples(
        examples=[
            ["living", "modern", PRESET_DETAILS["living"], 3, 30],
            ["dormitor", "scandinav", PRESET_DETAILS["dormitor"], 3, 30],
            ["dormitor 2", "scandinav", PRESET_DETAILS["dormitor 2"], 3, 30],
            ["bucătărie", "industrial", PRESET_DETAILS["bucătărie"], 4, 35],
        ],
        inputs=[room_type, style, custom_details, width_multiplier, num_steps],
    )

if __name__ == "__main__":
    demo.launch(share=True)