import torch
from diffusers import DiffusionPipeline
import time 

pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
)
pipe = pipe.to("cpu")

prompt = "a photo of an astronaut riding a horse on mars"
pipe.enable_attention_slicing()

start = time.time()
image = pipe(prompt).images[0]
end = time.time()
print("cost: ", end - start)


print(image)
image.save("./1.jpg")