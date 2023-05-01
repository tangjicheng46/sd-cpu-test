from optimum.onnxruntime import ORTStableDiffusionPipeline
import torch 
import time 

# model_id = "sd_v15_onnx"
model_path = "/home/ec2-user/SageMaker/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt"
pipe = ORTStableDiffusionPipeline.from_pretrained(model_path)

pipe.to("cuda")

torch.cuda.synchronize()
start = time.time()

prompt = "a photo of an astronaut riding a horse on mars"
images = pipe(prompt=prompt, num_inference_steps=20).images[0]

torch.cuda.synchronize()
end = time.time()
print("cost: ", end - start)