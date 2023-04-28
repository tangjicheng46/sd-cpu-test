from optimum.intel.openvino import OVStableDiffusionPipeline

model_id = "runwayml/stable-diffusion-v1-5"
pipe = OVStableDiffusionPipeline.from_pretrained(model_id, export=True)
prompt = "a photo of an astronaut riding a horse on mars"
images = pipe(prompt).images[0]