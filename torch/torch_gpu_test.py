import torch
from diffusers import DiffusionPipeline
import time 

model_path = "/home/ec2-user/SageMaker/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt"

pipe = DiffusionPipeline.from_pretrained(
    model_path,
)
pipe = pipe.to("cuda")

# pipe.save_pretrained("./model1")

prompt = "a photo of an astronaut riding a horse on mars"
# pipe.enable_attention_slicing()

torch.cuda.synchronize()
start = time.time()

image = pipe(prompt=prompt, num_inference_steps=20).images[0]

torch.cuda.synchronize()
end = time.time()
print("cost: ", end - start)


print(image)
image.save("./1.jpg")