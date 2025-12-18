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
    "bedroom": "bed, white furniture (elegant bookcase, wardrobe, desk, chair, nightstand), cozy",
    "bedroom 2": "bed, white furniture (elegant bookcase, wardrobe, TV), cozy",
    "living room": "light cream theme, L-shaped sofa, 2 armchairs, TV, elegant bookcase, window curtain",
    "kitchen": "modern appliances, marble countertops, island with bar stools, pendant lights"
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
with gr.Blocks(title="Room Panorama Generator") as demo:
    gr.Markdown("""
    # 🏠 Room Panorama Generator
    Generate cylindrical panoramic views of custom room designs without any API tokens!
    """)
    
    with gr.Row():
        with gr.Column():
            room_type = gr.Dropdown(
                choices=[
                    "bedroom",
                    "bedroom 2",
                    "living room",
                    "kitchen",
                    "bathroom",
                    "office",
                    "dining room",
                    "home theater",
                    "gym"
                ],
                value="living room",
                label="Room Type"
            )
            
            style = gr.Dropdown(
                choices=[
                    "modern",
                    "minimalist",
                    "industrial",
                    "scandinavian",
                    "traditional",
                    "contemporary",
                    "rustic",
                    "art deco",
                    "mid-century modern",
                    "japanese zen"
                ],
                value="modern",
                label="Design Style"
            )
            
            custom_details = gr.Textbox(
                label="Additional Details",
                placeholder="Auto-filled based on room type. You can edit this.",
                value=PRESET_DETAILS["living room"],
                lines=3
            )
            
            width_multiplier = gr.Slider(
                minimum=2,
                maximum=4,
                value=2,
                step=1,
                label="Panorama Width (multiplier)",
                info="Higher = wider panoramic view"
            )
            
            num_steps = gr.Slider(
                minimum=20,
                maximum=50,
                value=30,
                step=5,
                label="Generation Steps",
                info="Higher = better quality but slower"
            )
            
            generate_btn = gr.Button("🎨 Generate Panorama", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(
                label="Generated Panorama",
                type="pil"
            )
            
            gr.Markdown("""
            ### Tips:
            - Width multiplier 3-4 works best for panoramic views
            - Room presets auto-fill details (editable)
            - Add specific details for better results
            - Generation takes ~30-60 seconds on Colab GPU
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
            ["living room", "modern", PRESET_DETAILS["living room"], 3, 30],
            ["bedroom", "scandinavian", PRESET_DETAILS["bedroom"], 3, 30],
            ["bedroom 2", "scandinavian", PRESET_DETAILS["bedroom 2"], 3, 30],
            ["kitchen", "industrial", PRESET_DETAILS["kitchen"], 4, 35],
        ],
        inputs=[room_type, style, custom_details, width_multiplier, num_steps],
    )

if __name__ == "__main__":
    demo.launch(share=True)