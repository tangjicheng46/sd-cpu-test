from optimum.onnxruntime import ORTStableDiffusionPipeline

model_id = "sd_v15_onnx"
pipe = ORTStableDiffusionPipeline.from_pretrained(model_id)

pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
images = pipe(prompt=prompt, num_inference_steps=20).images[0]