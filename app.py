# from flask import Flask, request, render_template, send_from_directory
# import torch
# from diffusers import StableDiffusionPipeline

# app = Flask(__name__)

# # Load model for CPU
# pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
# pipe.to("cpu")
# pipe.enable_attention_slicing()
# pipe.enable_vae_slicing()
# pipe.safety_checker = None

# @app.route("/", methods=["GET", "POST"])
# def index():
#     image_url = None
    
#     if request.method == "POST":
#         prompt = request.form["prompt"]

#         # Generate image
#         image = pipe(prompt, num_inference_steps=30, height=512, width=512, guidance_scale=8.0).images[0]

#         # Save the image
#         image_path = "static/generated_image.png"
#         image.save(image_path)
#         image_url = "/" + image_path  

#     return render_template("index.html", image_url=image_url)

# # Route for serving static files
# @app.route('/static/generated_image.png')
# def serve_static(filename):
#     return send_from_directory('static', filename)

# if __name__ == "__main__":
#     app.run(debug=True)




from flask import Flask, request, render_template
from diffusers import StableDiffusionPipeline
import torch
import os

app = Flask(__name__)

# Load model for CPU
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe.to("cpu")
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.safety_checker = None

@app.route("/", methods=["GET", "POST"])
def generate_image():
    image_url = None
    if request.method == "POST":
        prompt = request.form["prompt"]
        image = pipe(
            prompt,
            num_inference_steps=30,
            height=512, width=512,
            guidance_scale=8.0
        ).images[0]

        image_path = "static/generated_image.png"
        os.makedirs("static", exist_ok=True)  # Create folder if not exists
        image.save(image_path)

        image_url = "/" + image_path

    return render_template("index.html", image_url=image_url)

if __name__ == "__main__":
    app.run(debug=True)

