from optimum.intel.openvino import OVStableDiffusionPipeline
import time


def run_vino_inference(prompt: str, is_save: bool = False):
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = OVStableDiffusionPipeline.from_pretrained(model_id, export=True)

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
            image.save("./vino_generated_image.jpg")
    print("Performance test is over.")
    print(f"Average cost is {total_cost_time / loop}")


if __name__ == "__main__":
    prompt = "a photo of an astronaut riding a horse on mars"

    run_vino_inference(prompt=prompt)
