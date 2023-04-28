from optimum.onnxruntime import ORTStableDiffusionPipeline
import time 

model_id = "sd_v15_onnx"
pipe = ORTStableDiffusionPipeline.from_pretrained(model_id)
prompt = "a photo of an astronaut riding a horse on mars"

start = time.time()
images = pipe(prompt=prompt, num_inference_steps=20).images[0]
end = time.time()
print("cost: ", end - start)