#!/usr/bin/env python3
"""
Generate a room layout visualization from JSON description
"""

import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

class RoomVisualizer:
    def __init__(self, json_data):
        self.room = json_data.get("room_description", {})
        self.fig = None
        self.ax = None
        
        # Color scheme based on Feng Shui style
        self.colors = {
            "background": "#F5F0E6",  # Cream/beige
            "wall": "#8B7355",        # Brown
            "door": "#8B4513",        # SaddleBrown
            "window": "#87CEEB",      # SkyBlue
            "furniture": "#D2B48C",   # Tan
            "text": "#2F4F4F",        # DarkSlateGray
            "plants": "#228B22",      # ForestGreen
            "bed": "#9370DB",         # MediumPurple
            "desk": "#DEB887",        # BurlyWood
            "wardrobe": "#BC8F8F",    # RosyBrown
            "bookcase": "#A0522D",    # Sienna
            "nightstand": "#CD853F"   # Peru
        }
        
    def align_to_coords(self, align_str, margin=0.1):
        """Convert alignment string to coordinates"""
        align_map = {
            "north": (0.5, 1 - margin),
            "north-northeast": (0.625, 1 - margin),
            "northeast": (0.75, 1 - margin),
            "east-northeast": (0.875, 0.75),
            "east": (1 - margin, 0.5),
            "east-southeast": (0.875, 0.25),
            "southeast": (0.75, margin),
            "south-southeast": (0.625, margin),
            "south": (0.5, margin),
            "south-southwest": (0.375, margin),
            "southwest": (0.25, margin),
            "west-southwest": (0.125, 0.25),
            "west": (margin, 0.5),
            "west-northwest": (0.125, 0.75),
            "northwest": (0.25, 1 - margin)
        }
        return align_map.get(align_str.lower(), (0.5, 0.5))
    
    def draw_room(self):
        """Draw the room outline"""
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        
        # Set background
        self.fig.patch.set_facecolor(self.colors["background"])
        self.ax.set_facecolor(self.colors["background"])
        
        # Draw walls
        wall = patches.Rectangle((0.05, 0.05), 0.9, 0.9, 
                                linewidth=3, 
                                edgecolor=self.colors["wall"],
                                facecolor='none')
        self.ax.add_patch(wall)
        
        # Draw center point (Feng Shui center)
        self.ax.plot(0.5, 0.5, 'o', color='gold', markersize=20, 
                    alpha=0.3, label='Center (Feng Shui)')
        
    def draw_item(self, item):
        """Draw a room item"""
        if item.get("display") != "block":
            return
            
        item_type = item.get("type", "").lower()
        align = item.get("align", "center")
        x, y = self.align_to_coords(align)
        
        # Adjust size based on item type
        size_map = {
            "bed": 0.15,
            "wardrobe": 0.12,
            "desk": 0.10,
            "bookcase": 0.12,
            "nightstand": 0.08,
            "door": 0.08,
            "window": 0.10
        }
        
        size = size_map.get(item_type, 0.1)
        
        # Draw different items with different shapes
        if item_type == "bed":
            # Draw bed as rectangle with pillow
            bed = patches.Rectangle((x - size/2, y - size/4), size, size/2,
                                  linewidth=2, 
                                  edgecolor=self.colors["wall"],
                                  facecolor=self.colors["bed"],
                                  alpha=0.8)
            self.ax.add_patch(bed)
            # Pillow
            pillow = patches.Ellipse((x - size/3, y), size/4, size/6,
                                   facecolor='white', edgecolor='gray')
            self.ax.add_patch(pillow)
            
        elif item_type == "wardrobe":
            # Draw wardrobe
            wardrobe = patches.Rectangle((x - size/2, y - size/2), size, size,
                                       linewidth=2,
                                       edgecolor=self.colors["wall"],
                                       facecolor=self.colors["wardrobe"],
                                       alpha=0.8)
            self.ax.add_patch(wardrobe)
            # Draw doors
            self.ax.plot([x - size/4, x - size/4], [y - size/2, y + size/2], 
                        'k-', linewidth=2)
            self.ax.plot([x + size/4, x + size/4], [y - size/2, y + size/2], 
                        'k-', linewidth=2)
            
        elif item_type == "desk":
            # Draw desk
            desk = patches.Rectangle((x - size, y - size/3), size*2, size/1.5,
                                   linewidth=2,
                                   edgecolor=self.colors["wall"],
                                   facecolor=self.colors["desk"],
                                   alpha=0.8)
            self.ax.add_patch(desk)
            
            if item.get("chair"):
                # Draw chair
                chair = patches.Rectangle((x, y - size/1.5), size/2, size/2,
                                        linewidth=1,
                                        edgecolor=self.colors["wall"],
                                        facecolor=self.colors["furniture"])
                self.ax.add_patch(chair)
                
        elif item_type == "bookcase":
            # Draw bookcase
            bookcase = patches.Rectangle((x - size/2, y - size/2), size, size,
                                       linewidth=2,
                                       edgecolor=self.colors["wall"],
                                       facecolor=self.colors["bookcase"],
                                       alpha=0.8)
            self.ax.add_patch(bookcase)
            # Draw shelves
            for i in range(1, 4):
                shelf_y = y - size/2 + i * size/4
                self.ax.plot([x - size/2, x + size/2], [shelf_y, shelf_y], 
                            'k-', linewidth=1)
                
        elif item_type == "nightstand":
            # Draw nightstand
            nightstand = patches.Rectangle((x - size/2, y - size/2), size, size,
                                         linewidth=2,
                                         edgecolor=self.colors["wall"],
                                         facecolor=self.colors["nightstand"],
                                         alpha=0.8)
            self.ax.add_patch(nightstand)
            # Draw lamp
            self.ax.plot(x, y, 'o', color='yellow', markersize=size*50,
                        alpha=0.7)
            
        elif item_type == "door":
            # Draw door
            door = patches.Rectangle((x - size/3, y - size/2), size/1.5, size,
                                   linewidth=2,
                                   edgecolor=self.colors["door"],
                                   facecolor='#F5DEB3',
                                   alpha=0.8)
            self.ax.add_patch(door)
            
        elif item_type == "window":
            # Draw window
            window = patches.Rectangle((x - size, y - size/3), size*2, size/1.5,
                                     linewidth=2,
                                     edgecolor=self.colors["window"],
                                     facecolor='#E0FFFF',
                                     alpha=0.6)
            self.ax.add_patch(window)
            # Window panes
            self.ax.plot([x, x], [y - size/3, y + size/3], 'k-', linewidth=1)
            self.ax.plot([x - size, x + size], [y, y], 'k-', linewidth=1)
            
            if item.get("plants"):
                # Draw plants
                for i in range(3):
                    plant_x = x - size/2 + i * size/2
                    plant = patches.Circle((plant_x, y - size/2), size/6,
                                         facecolor=self.colors["plants"])
                    self.ax.add_patch(plant)
                    
        # Add label
        self.ax.text(x, y + size*0.8, item.get("type", ""), 
                    ha='center', va='bottom',
                    fontsize=9, color=self.colors["text"],
                    bbox=dict(boxstyle="round,pad=0.3", 
                             facecolor="white", 
                             alpha=0.7))
    
    def draw_compass(self):
        """Draw a compass rose"""
        compass_size = 0.15
        compass_x, compass_y = 0.1, 0.9
        
        # Draw compass circle
        compass = patches.Circle((compass_x, compass_y), compass_size/2,
                               facecolor='white', edgecolor='black', alpha=0.8)
        self.ax.add_patch(compass)
        
        # Directions
        directions = ['N', 'E', 'S', 'W']
        positions = [(compass_x, compass_y + compass_size/3),
                    (compass_x + compass_size/3, compass_y),
                    (compass_x, compass_y - compass_size/3),
                    (compass_x - compass_size/3, compass_y)]
        
        for direction, (pos_x, pos_y) in zip(directions, positions):
            self.ax.text(pos_x, pos_y, direction, 
                        ha='center', va='center',
                        fontsize=10, fontweight='bold',
                        color='red')
        
        # Arrow for north
        self.ax.arrow(compass_x, compass_y - compass_size/6,
                     0, compass_size/3,
                     head_width=0.02, head_length=0.02,
                     fc='red', ec='red')
    
    def add_title(self):
        """Add title and room information"""
        title = f"{self.room.get('name', 'Room')}\n{self.room.get('description', '')}"
        dims = self.room.get('dimensions', [0, 0, 0])
        dim_text = f"Dimensions: {dims[0]}m × {dims[1]}m × {dims[2]}m"
        style_text = f"Style: {self.room.get('style', '')}"
        
        self.ax.set_title(title, fontsize=14, fontweight='bold', 
                         color=self.colors["text"], pad=20)
        
        # Add info box
        info_text = f"{dim_text}\n{style_text}\nID: {self.room.get('id', '')}"
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        self.ax.text(0.02, 0.02, info_text, transform=self.ax.transAxes,
                    fontsize=9, verticalalignment='bottom',
                    bbox=props, color=self.colors["text"])
    
    def generate(self, output_path):
        """Generate the complete visualization"""
        self.draw_room()
        
        # Draw all items
        for item in self.room.get("items", []):
            self.draw_item(item)
        
        self.draw_compass()
        self.add_title()
        
        # Remove axis
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Image saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate room layout visualization')
    parser.add_argument('--input', '-i', default='room_description.json',
                       help='Input JSON file')
    parser.add_argument('--output', '-o', default='room_layout.png',
                       help='Output image file')
    
    args = parser.parse_args()
    
    # Load JSON data
    try:
        with open(args.input, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found.")
        print("Creating example JSON file...")
        
        # Create example JSON
        example_data = {
            "room_description": {
                "id": "room_20251209_001",
                "name": "Bedroom 001",
                "description": "User's bedroom",
                "shape": "rectangular",
                "dimensions": [3.2, 3.1, 2.2],
                "style": "Feng Shui",
                "items": [
                    {"type": "Door", "align": "west-southwest", "display": "none"},
                    {"type": "Bookcase", "align": "west-northwest", "display": "none"},
                    {"type": "Wardrobe", "align": "north", "display": "block"},
                    {"type": "Bed", "align": "north-northeast", "display": "block"},
                    {"type": "Window", "plants": True, "align": "east", "display": "block"},
                    {"type": "Desk", "chair": True, "align": "south-southeast", "display": "block"},
                    {"type": "Nightstand", "align": "south", "display": "block"}
                ]
            }
        }
        
        with open('room_description.json', 'w') as f:
            json.dump(example_data, f, indent=2)
        
        data = example_data
        print("Example JSON file created: room_description.json")
    
    # Generate visualization
    visualizer = RoomVisualizer(data)
    visualizer.generate(args.output)

if __name__ == "__main__":
    main()