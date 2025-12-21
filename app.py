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
    "dormitor": """
{
	"room_description": {
		"id": "room_20251215_001",
		"name": "Bedroom 001",
		"description": "User's bedroom",
		"shape": "rectangular",
		"dimensions": [ 3.2, 3.1, 2.2 ],
		"style": "Romanian Creative",
      "theme": "light cream",
      "requirement": "there should be enough furniture in the room",
		"items": {
			{
				"type": "Door",
				"align": "west-southwest"
			},
			{
				"type": "Bookcase",
				"align": "west-northwest",
            "style": "decent"
			},
			{
				"type": "Wardrobe",
				"align": "north"
			},
			{
				"type": "Bed",
				"align": "north-northeast, towards room"
			},
			{
				"type": "Classical Window",
				"type": "Plants",
				"align": "east",
            "accessories": "curtain"
			},
			{
				"type": "Desk",
				"type": "Chair",
				"align": "south-southeast"
			},
			{
				"type": "Nightstand",
				"align": "south"
			}
		}
	},
	"role": "image generator",
	"task": "generate a 400 degrees cylindric panorama of room_20251215_001."
}
    """,
    "dormitor 2": """
{
	"room_description": {
		"id": "room_20251215_002",
		"name": "Bedroom 002",
		"description": "User's bedroom",
		"shape": "rectangular",
		"dimensions": [3.0, 3.0, 2.2],
		"style": "Romanian Creative",
      "theme": "light cream",
      "requirement" :"Bookcase and Wardrobe should be a U-form above the bed",
		"items": [
			{
				"type": "Door",
				"align": "west-southwest",
             "radius": 0.9
			},
			{
				"type": "Bookcase",
				"align": "north-northwest; north above the bed",
             "style": "elegant"
			},
			{
				"type": "Wardrobe",
				"align": "north-northeast"
			},
			{
				"type": "Bed",
				"align": "north, under the other furniture"
			},
         {
            "type": "TV",
            "align": "south, fixed in the wall"
         },
			{
				"type": "Classical Window",
				"align": "east",
            "accesories": "curtain"
			}
		]
	},
	"role": "image generator",
	"task": "generate a 400 degrees cylindrical panorama of room_20251215_002."
}
    """,
    "living": """
{
  "room_description": {
    "id": "living_20251215_003",
    "name": "Living 003",
    "description": "User's living",
    "shape": "rectangular, with an arch of 3m near west door, 60 degrees at west-southwest",
    "dimensions": [5.0, 4.0, 2.2],
    "style": "Creative",
    "theme": "light cream",
    "optimization": "maximize amount of furniture",
    "items": [
      {
        "type": "Door",
        "align": ["west", "north-northwest"],
        "count": 2
      },
      {
        "type": "Affordable L-shaped sofa",
        "align": "fitting the arch",
        "sizes": [2.0, 1.2]
      },
      {
        "type": "TV",
        "align": "creative"
      },
      {
        "type": "Traditional Window",
        "align": "east",
        "position": "centered",
        "total_width": 1.0,
        "accessories": "curtain"
      },
      {
        "type": "Affordable Armchair",
        "align": "creative",
        "count": 2
      },
      {
        "type": "Bookcase",
        "align": "creative",
        "style": "elegant",
        "length": 3.0
      }
    ]
  },
  "role": "image generator",
  "task": "generate a 400 degrees cylindric panorama of living_20251215_003"
}
    """,
    #"temă crem deschis, canapea în L, 2 fotolii, TV, bibliotecă elegantă, perdele",
    "bucătărie": """
    {
  "room_description": {
    "id": "kitchen_20251220_004",
    "name": "Kitchen 004",
    "description": "User's kitchen",
    "shape": "rectangular",
    "dimensions": [7.0, 5.0, 2.2],
    "style": "Romanian Creative",
    "theme": "light cream",
    "items": [
      {
        "type": "Door",
        "align": ["west", "south-southeast"],
        "count": 2
      },
      {
        "type": "Classical Window",
        "align": "east",
        "position": "centered",
        "total_width": 1.0,
      },
      {
			"type": "usual Romanian-style Kitchen furniture",
			"placement": "creative",
          "theme": "light cream"
		},
    ]
  },
  "role": "image generator",
  "task": "generate a 400 degrees cylindric panorama of kitchen_20251220_004"
}
    """,
    "baie": """
{
  "room_description": {
    "id": "bathroom_20251220_005",
    "name": "Bathroom 005",
    "description": "User's bathroom",
    "shape": "rectangular",
    "dimensions": [4.5, 2.5, 2.2],
    "style": "Creative Romanian",
    "theme": "light cream",
    "items": [
      {
        "type": "Door",
        "align": "north-northwest",
      },
      {
        "type": "Classical Window",
        "align": "east-southeast",
      },
		{
			"type": "Bath",
          "align": south"
		},
		{
			"type": "Sink",
			"align": "west"
		},
		{
			"type": "WC",
			"align": "west-northwest"
		}
    ]
  },
  "role": "image generator",
  "task": "generate a 400 degrees cylindric panorama of bathroom_20251220_005"
}
    """
}


def update_custom_details(room_type):
    """Update custom_details textbox based on selected room_type"""
    return PRESET_DETAILS.get(room_type, "")

def generate_room_panorama(room_type, style, custom_details, width_multiplier, num_steps):
    # Create panoramic prompt
    base_prompt = f"{room_type.replace(' 2', '')} interior design"
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