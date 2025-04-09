import os
import requests
from datetime import datetime
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure X.AI client
XAI_API_KEY = os.getenv("GROK_API_KEY")
client = OpenAI(base_url="https://api.x.ai/v1", api_key=XAI_API_KEY)

def generate_contextual_image(prompt, filename_prefix):
    """Generate image using X.AI's grok-2-image model."""
    print(f"\n🎨 Generating image for: {filename_prefix}")
    print(f"Prompt: {prompt}")
    
    try:
        # Request image from X.AI
        response = client.images.generate(
            model="grok-2-image-1212",
            prompt=prompt,
            n=1
        )
        
        # Get the image URL
        image_url = response.data[0].url
        print(f"✅ Image URL: {image_url}")
        
        # Download the image
        img_response = requests.get(image_url)
        if img_response.status_code == 200:
            # Create timestamped filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            image_path = f"{filename_prefix}_{timestamp}.png"
            
            # Save the image
            img = Image.open(BytesIO(img_response.content))
            img.save(image_path)
            print(f"✅ Image saved to: {image_path}")
            
            return image_path, "AI-generated image"
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    return None, None

def test_crime_contextual_images():
    """Generate contextual images for crime analysis report."""
    print("\n🔍 Generating crime analysis contextual images...")
    
    # Define regions for context
    regions = ["Chicago", "New York"]
    regions_str = ", ".join(regions)
    
    # Define prompts for different contextual images
    prompts = [
        {
            "title": "Crime Prevention Strategies",
            "prompt": f"A photorealistic image showing modern crime prevention strategies in {regions_str}. Community watch programs, police presence, and advanced surveillance technology working together. Professional style for a data report.",
            "prefix": "crime_prevention"
        },
        {
            "title": f"Urban Safety in {regions[0]}",
            "prompt": f"A photorealistic image of urban safety features in {regions[0]}. Well-lit streets, security cameras, police patrols, and people feeling safe in public spaces. Professional style for a crime analysis report.",
            "prefix": "urban_safety"
        },
        {
            "title": "Community Policing Impact",
            "prompt": "A photorealistic image of effective community policing. Police officers interacting positively with diverse community members, participating in neighborhood events, and building trust. Professional style for a crime analysis report.",
            "prefix": "community_policing"
        }
    ]
    
    # Generate images for each prompt
    contextual_images = {}
    
    for prompt_data in prompts:
        image_path, description = generate_contextual_image(
            prompt_data["prompt"],
            prompt_data["prefix"]
        )
        
        if image_path:
            contextual_images[prompt_data["title"]] = {
                "path": image_path,
                "prompt": prompt_data["prompt"],
                "description": description,
                "rationale": f"Illustrative image for {prompt_data['title']}"
            }
    
    # Print summary
    print(f"\n✅ Generated {len(contextual_images)} contextual images")
    for title, data in contextual_images.items():
        print(f"- {title}: {data['path']}")
    
    # Display images
    for title, data in contextual_images.items():
        try:
            img = Image.open(data["path"])
            plt.figure(figsize=(10, 6))
            plt.imshow(img)
            plt.title(title)
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"❌ Error displaying image: {str(e)}")
    
    return contextual_images

if __name__ == "__main__":
    # Test the image generation
    contextual_images = test_crime_contextual_images()