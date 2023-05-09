from optimum.intel.openvino import OVStableDiffusionPipeline
import time

model_id = "runwayml/stable-diffusion-v1-5"
pipe = OVStableDiffusionPipeline.from_pretrained(model_id)
prompt = "a photo of an astronaut riding a horse on mars"

start = time.time()
images = pipe(prompt=prompt, num_inference_steps=20).images[0]
images.save("./1.jpg")
end = time.time()

print("cost: ", end - start)