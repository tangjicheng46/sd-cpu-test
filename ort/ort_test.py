from optimum.onnxruntime import ORTStableDiffusionPipeline

model_id = "sd_v15_onnx"
pipe = ORTStableDiffusionPipeline.from_pretrained(model_id)
prompt = "a photo of an astronaut riding a horse on mars"
images = pipe(prompt).images[0]