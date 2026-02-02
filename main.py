import argparse
import torch
from pipeline_flux_rf_inversion import FluxRFInversionPipeline
from diffusers import FluxImg2ImgPipeline
from diffusers.utils import load_image
import os
import numpy as np
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Run Flux Pipeline")
    parser.add_argument("--model", type=str, default="/path/to/FLUX1-dev", help="Model name or path")
    parser.add_argument("--image", type=str, default="https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg", help="URL of the input image")
    parser.add_argument(
        "--prompt_list", 
        nargs='+', 
        default=None,
        help="List of prompts"
    )
    parser.add_argument("--prompt_2", type=str, help="Second prompt (if different from prompt)")
    parser.add_argument("--num_inference_steps", type=int, default=28, help="Number of inference steps")
    parser.add_argument("--strength", type=float, default=0.95, help="Strength parameter")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale")
    parser.add_argument("--gamma", type=float, default=0.5, help="Gamma parameter")
    parser.add_argument("--eta", type=float, default=0.9, help="Eta parameter")
    parser.add_argument("--start_timestep", type=int, default=0, help="Start timestep")
    parser.add_argument("--stop_timestep", type=int, default=6, help="Stop timestep")
    parser.add_argument("--output", type=str, default="output.jpg", help="Output image filename")
    parser.add_argument("--use_img2img", action="store_true", help="Use FluxImg2ImgPipeline")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.use_img2img:
        pipe = FluxImg2ImgPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    else:
        pipe = FluxRFInversionPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    
    pipe = pipe.to(device)

    if args.image.startswith('http://') or args.image.startswith('https://'):
        init_image = load_image(args.image).resize((1024, 1024))
    else:
        file_name = os.path.splitext(os.path.basename(args.image))[0]
        pt_file_path = f"/{file_name}_latents.pt"
        if os.path.exists(pt_file_path):
            init_image = args.image
        else:
            init_image = args.image
    block_id_list = [[0]]

    
    prompt_list = args.prompt_list
    kwargs = {"gamma": 0.5, "eta": 0.9, "start_timestep": 0, "stop_timestep": 9},
    torch.manual_seed(8889)
    save_path = "/save_dir"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for block_id in block_id_list:
        images = pipe(
            prompt=prompt_list,
            image=init_image,
            num_inference_steps=args.num_inference_steps,
            strength=args.strength,
            generator=torch.Generator("cpu").manual_seed(8889),
            guidance_scale=args.guidance_scale,
            block_id = block_id,
            **kwargs,
        ).images
        joined_images = np.concatenate(images, axis=1)
        img_path="res.png"
        Image.fromarray(joined_images).save(os.path.join(save_path, img_path))
if __name__ == "__main__":
    main()