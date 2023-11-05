#!/usr/bin/env python

from __future__ import annotations
from diffusers.utils import load_image
import os
import random

import cv2
import gradio as gr
import numpy as np
import PIL.Image
from PIL import Image
import torch
# from diffusers import AutoencoderKL, StableDiffusionXLPipeline
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL, DiffusionPipeline
import uuid

DESCRIPTION = '''# Segmind Stable Diffusion: SSD-1B
#### [Segmind's SSD-1B](https://huggingface.co/segmind/SSD-1B) is a distilled, 50% smaller version of SDXL, offering up to 60% speedup
'''
if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"

MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES", "1") == "1"
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "1024"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "1") == "0"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"
ENABLE_REFINER = os.getenv("ENABLE_REFINER", "1") == "1"

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0")
style_list = [
    {
        "name": "(No style)",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Cinematic",
        "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    },
    {
        "name": "Photographic",
        "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
        "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
    },
    {
        "name": "Anime",
        "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed",
        "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast",
    },
    {
        "name": "Manga",
        "prompt": "manga style {prompt} . vibrant, high-energy, detailed, iconic, Japanese comic style",
        "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, Western comic style",
    },
    {
        "name": "Digital Art",
        "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
        "negative_prompt": "photo, photorealistic, realism, ugly",
    },
    {
        "name": "Pixel art",
        "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics",
        "negative_prompt": "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic",
    },
    {
        "name": "Fantasy art",
        "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
        "negative_prompt": "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white",
    },
    {
        "name": "Neonpunk",
        "prompt": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional",
        "negative_prompt": "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
    },
    {
        "name": "3D Model",
        "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
        "negative_prompt": "ugly, deformed, noisy, low poly, blurry, painting",
    },
]

styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Cinematic"


def apply_style(style_name: str, positive: str, negative: str = "") -> Tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    if not negative:
        negative = ""
    return p.replace("{prompt}", positive), n + negative


if torch.cuda.is_available():

    # initialize the models and pipeline
    controlnet_conditioning_scale = 0.5  # recommended for good generalization
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
    )
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "segmind/SSD-1B", controlnet=controlnet, vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    # pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    #     "/data/SSD-1B", controlnet=controlnet, vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    # )

    # vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    # pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    #    "/data/SSD-1B",
    #    vae=vae,
    #    torch_dtype=torch.float16,
    #    use_safetensors=True,
    #    variant="fp16",
    # )
    if ENABLE_REFINER:
        refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            vae=vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )

    if ENABLE_CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()
        if ENABLE_REFINER:
            refiner.enable_model_cpu_offload()
    else:
        pipe.to(device)
        if ENABLE_REFINER:
            refiner.to(device)
        print("Loaded on Device!")

    if USE_TORCH_COMPILE:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        if ENABLE_REFINER:
            refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)
        print("Model Compiled!")


def save_image(img):
    unique_name = str(uuid.uuid4()) + '.png'
    img.save(unique_name)
    return unique_name


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def generate(
        prompt: str,
        img: Image,
        negative_prompt: str = "",
        style: str = DEFAULT_STYLE_NAME,
        prompt_2: str = "",
        negative_prompt_2: str = "",
        use_negative_prompt: bool = False,
        use_prompt_2: bool = False,
        use_negative_prompt_2: bool = False,
        seed: int = 0,
        width: int = 1024,
        height: int = 1024,
        guidance_scale_base: float = 5.0,
        guidance_scale_refiner: float = 5.0,
        num_inference_steps_base: int = 25,
        num_inference_steps_refiner: int = 25,
        apply_refiner: bool = False,
        randomize_seed: bool = False,
        progress=gr.Progress(track_tqdm=True)
):
    seed = randomize_seed_fn(seed, randomize_seed)
    generator = torch.Generator().manual_seed(seed)
    image = img
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    if not use_negative_prompt:
        negative_prompt = None  # type: ignore
    if not use_prompt_2:
        prompt_2 = None  # type: ignore
    if not use_negative_prompt_2:
        negative_prompt_2 = None  # type: ignore
    prompt, negative_prompt = apply_style(style, prompt, negative_prompt)
    if not apply_refiner:
        image = pipe(
            prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, image=canny_image).images[0]
        # image = pipe(
        #    prompt=prompt,
        #    negative_prompt=negative_prompt,
        #    prompt_2=prompt_2,
        #    negative_prompt_2=negative_prompt_2,
    #     width=width,
    #     height=height,
    #     guidance_scale=guidance_scale_base,
    #     num_inference_steps=num_inference_steps_base,
    #    generator=generator,
    #    output_type="pil",
    # ).images[0]
    else:
        latents = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_2=prompt_2,
            negative_prompt_2=negative_prompt_2,
            width=width,
            height=height,
            guidance_scale=guidance_scale_base,
            num_inference_steps=num_inference_steps_base,
            generator=generator,
            output_type="latent",
        ).images
        image = refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_2=prompt_2,
            negative_prompt_2=negative_prompt_2,
            guidance_scale=guidance_scale_refiner,
            num_inference_steps=num_inference_steps_refiner,
            image=latents,
            generator=generator,
        ).images[0]

    image_path = save_image(image)
    print(image_path)
    return [image_path], seed


examples = [
    '3d digital art of an adorable ghost, glowing within, holding a heart shaped pumpkin, Halloween, super cute, spooky haunted house background',
    'beautiful lady, freckles, big smile, blue eyes, short ginger hair, dark makeup, wearing a floral blue vest top, soft light, dark grey background',
    'professional portrait photo of an anthropomorphic cat wearing fancy gentleman hat and jacket walking in autumn forest.',
    'an astronaut sitting in a diner, eating fries, cinematic, analog film',
    'Albert Einstein in a surrealist Cyberpunk 2077 world, hyperrealistic',
    'cinematic film still of Futuristic hero with golden dark armour with machine gun,  muscular body']

with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(
        value="Duplicate Space for private use",
        elem_id="duplicate-button",
        visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
    )
    with gr.Group():
        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
            img = gr.Image(type="pil")
            run_button = gr.Button("Run", scale=0)
        result = gr.Gallery(label="Result", columns=1, show_label=False)
    with gr.Accordion("Advanced options", open=False):
        with gr.Row():
            use_negative_prompt = gr.Checkbox(label="Use negative prompt", value=False)
            use_prompt_2 = gr.Checkbox(label="Use prompt 2", value=False)
            use_negative_prompt_2 = gr.Checkbox(label="Use negative prompt 2", value=False)
        style_selection = gr.Radio(
            show_label=True, container=True, interactive=True,
            choices=STYLE_NAMES,
            value=DEFAULT_STYLE_NAME,
            label='Image Style'
        )
        negative_prompt = gr.Text(
            label="Negative prompt",
            max_lines=1,
            placeholder="Enter a negative prompt",
            visible=False,
        )
        prompt_2 = gr.Text(
            label="Prompt 2",
            max_lines=1,
            placeholder="Enter your prompt",
            visible=False,
        )
        negative_prompt_2 = gr.Text(
            label="Negative prompt 2",
            max_lines=1,
            placeholder="Enter a negative prompt",
            visible=False,
        )
        seed = gr.Slider(
            label="Seed",
            minimum=0,
            maximum=MAX_SEED,
            step=1,
            value=0,
        )
        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
        with gr.Row(visible=False):
            width = gr.Slider(
                label="Width",
                minimum=256,
                maximum=MAX_IMAGE_SIZE,
                step=32,
                value=1024,
            )
            height = gr.Slider(
                label="Height",
                minimum=256,
                maximum=MAX_IMAGE_SIZE,
                step=32,
                value=1024,
            )
        apply_refiner = gr.Checkbox(label="Apply refiner", value=False, visible=ENABLE_REFINER)
        with gr.Row():
            guidance_scale_base = gr.Slider(
                label="Guidance scale for base",
                minimum=1,
                maximum=20,
                step=0.1,
                value=9.0,
            )
            num_inference_steps_base = gr.Slider(
                label="Number of inference steps for base",
                minimum=10,
                maximum=100,
                step=1,
                value=25,
            )
        with gr.Row(visible=False) as refiner_params:
            guidance_scale_refiner = gr.Slider(
                label="Guidance scale for refiner",
                minimum=1,
                maximum=20,
                step=0.1,
                value=5.0,
            )
            num_inference_steps_refiner = gr.Slider(
                label="Number of inference steps for refiner",
                minimum=10,
                maximum=100,
                step=1,
                value=25,
            )

    gr.Examples(
        examples=examples,
        inputs=prompt,
        outputs=[result, seed],
        fn=generate,
        cache_examples=CACHE_EXAMPLES,
    )

    use_negative_prompt.change(
        fn=lambda x: gr.update(visible=x),
        inputs=use_negative_prompt,
        outputs=negative_prompt,
        queue=False,
        api_name=False,
    )
    use_prompt_2.change(
        fn=lambda x: gr.update(visible=x),
        inputs=use_prompt_2,
        outputs=prompt_2,
        queue=False,
        api_name=False,
    )
    use_negative_prompt_2.change(
        fn=lambda x: gr.update(visible=x),
        inputs=use_negative_prompt_2,
        outputs=negative_prompt_2,
        queue=False,
        api_name=False,
    )
    apply_refiner.change(
        fn=lambda x: gr.update(visible=x),
        inputs=apply_refiner,
        outputs=refiner_params,
        queue=False,
        api_name=False,
    )

    gr.on(
        triggers=[
            prompt.submit,
            negative_prompt.submit,
            prompt_2.submit,
            negative_prompt_2.submit,
            run_button.click,
            img.upload,
        ],
        fn=generate,
        inputs=[
            prompt,
            img,
            negative_prompt,
            style_selection,
            prompt_2,
            negative_prompt_2,
            use_negative_prompt,
            use_prompt_2,
            use_negative_prompt_2,
            seed,
            width,
            height,
            guidance_scale_base,
            guidance_scale_refiner,
            num_inference_steps_base,
            num_inference_steps_refiner,
            apply_refiner,
            randomize_seed
        ],
        outputs=[result, seed],
        api_name="run",
    )

if __name__ == "__main__":
    demo.queue().launch(server_name='0.0.0.0', share=True)
