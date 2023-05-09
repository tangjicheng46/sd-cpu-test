import time
import argparse

import torch
from diffusers import StableDiffusionPipeline


def run_torch_inference(prompt: str, is_enable_slicing: bool = True, is_save: bool = False):
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    if is_enable_slicing:
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()

    # warmup
    image = pipe(prompt=prompt, num_inference_steps=20).images[0]

    # run loop for performance testing
    loop = 2
    total_cost_time = 0.0
    for i in range(loop):
        start = time.time()
        image = pipe(prompt=prompt, num_inference_steps=20).images[0]
        end = time.time()
        cost_time = end - start
        total_cost_time += cost_time
        print(f"[{i}] cost: {cost_time}")
        if is_save:
            image.save("./torch_generated_image.jpg")
    print("Performance test is over.")
    print(f"Average cost is {total_cost_time / loop}")


if __name__ == "__main__":
    prompt = "a photo of an astronaut riding a horse on mars"

    parser = argparse.ArgumentParser(prog="Stable Diffusion Performance Test")
    parser.add_argument("--enable_slicing", default=True, type=bool)

    args = parser.parse_args()

    run_torch_inference(prompt=prompt, is_enable_slicing=args.enable_slicing)
