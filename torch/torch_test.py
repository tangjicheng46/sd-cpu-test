import time
import argparse

import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline


def run_inference(model_path: str,
                  prompt: str,
                  device: str,
                  is_save: bool = True):
    pipe = StableDiffusionPipeline.from_pretrained(model_path)
    pipe = pipe.to(device)

    torch.cuda.synchronize()
    start = time.time()

    # inference
    image = pipe(prompt=prompt, num_inference_steps=20).images[0]

    torch.cuda.synchronize()
    end = time.time()
    print("cost: ", end - start)
    print(image)
    if is_save:
        image.save("./1.jpg")


if __name__ == "__main__":
    prompt = "a photo of an astronaut riding a horse on mars"

    parser = argparse.ArgumentParser(prog="Stable Diffusion Performance Test")

    parser.add_argument(
        "--model_path",
        default="../stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt",
        type=str)
    parser.add_argument("--device", default="cpu", type=str)

    args = parser.parse_args()

    run_inference(model_path=args.model_path,
                  prompt=prompt,
                  device=args.device)
