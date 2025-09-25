from PIL import Image, ImageDraw, ImageFont

import os, json
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

def parse_json(json_output):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output

def visualize_result(img, parsed_json, out_folder):
    draw = ImageDraw.Draw(img)

    # Suppose your points are in [y, x] format normalized to 0–1000

    points = [p['point'] for p in parsed_json]
    labels = [p['label'] for p in parsed_json]

    # Image size
    H, W = img.size[1], img.size[0]

    font = ImageFont.load_default()

    # Convert normalized points to image coordinates
    scaled_points = [(int(x / 1000 * W), int(y / 1000 * H)) for y, x in points]

    # Draw circles at each point
    for label, pt in zip(labels, scaled_points):
        r = 5  # radius
        draw.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], fill='red')
        draw.text((pt[0] + 6, pt[1] - 6), label, fill="white", font=font)
    
    img.save(os.path.join(out_folder, "points_gemini_red_blue2.png"))

if __name__ == "__main__":

    out_folder = '/home/saumyas/Projects/semnav/explore-eqa_semnav/outputs/media'
    img = Image.open('/home/saumyas/Projects/semnav/explore-eqa_semnav/outputs/media/red_blue2.png')

    # img = img.crop((750, 100, 1900, 1100)) # [100:1100,750:1900]
    # img = img.crop((0, 100, 1149, 900)) # [100:1100,750:1900]
    # img.save(os.path.join(out_folder, "cropped_img.jpg"))

    labels = f"""
            Blue box top left corner,
            Blue box top right corner,
            Blue box bottom left corner,
            Blue box bottom right corner,
            Red box top left corner,
            Red box top right corner,
            Red box bottom left corner,
            Red box bottom right corner,
            center point of ball, 
            center point of bowl, 
            center point of loofah, 
            silver handle (point on handle)
        """

    # Analyze the image using Gemini
    response = client.models.generate_content(
        model="gemini-2.5-pro-preview-03-25",
        contents=[
            img,
            f"""
            Point to objects in the list {labels}.
            The answer should follow the json format: ['point': <point>, 'label': <label1>, ...]. 
            The points are in [y, x] format normalized to 0-1000.
            """
        ],
        config = types.GenerateContentConfig(
            temperature=0.2
        )
    )
    # Also point to the silver handle attached to one of the boxes, corresponding to the point where it should be grasped to pick it up.
    
    # Check response
    print(response.text)
    parsed_json = parse_json(response.text)
    print(parsed_json)

    visualize_result(img, json.loads(parsed_json), out_folder)




