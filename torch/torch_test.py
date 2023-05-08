import time
import argparse

import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from optimum.onnxruntime import ORTStableDiffusionPipeline


def run_inference(pipe, prompt: str, device: str, is_save: bool = True):
    pipe = pipe.to(device)
    start = time.time()
    image = pipe(prompt=prompt, num_inference_steps=20).images[0]
    end = time.time()
    print("cost: ", end - start)
    print(image)
    if is_save:
        image.save("./1.jpg")


if __name__ == "__main__":
    prompt = "a photo of an astronaut riding a horse on mars"

    parser = argparse.ArgumentParser(prog="Stable Diffusion Performance Test")
    parser.add_argument("--model_path",
                        default="runwayml/stable-diffusion-v1-5",
                        type=str)
    parser.add_argument("--engine", default="torch", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--dtype", default="fp32", type=str)

    args = parser.parse_args()

    infer_dtype = torch.float32
    if args.dtype == "fp16":
        infer_dtype = torch.float16

    pipe = None
    if args.engine == "ort":
        pipe = ORTStableDiffusionPipeline.from_pretrained(
            args.model_path, torch_dtype=infer_dtype)
        pipe.save_pretrained("./my_onnx")
    else:
        pipe = StableDiffusionPipeline.from_pretrained(args.model_path,
                                                       torch_dtype=infer_dtype)
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        pipe.save_pretrained("./my_torch")

    N = 2
    for i in range(N):
        run_inference(pipe, prompt=prompt, device=args.device)
