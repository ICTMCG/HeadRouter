"""HeadRouter inference entry point (RF-Inversion + HeadRouter on FLUX)."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pipeline_flux_rf_inversion import FluxRFInversionPipeline

ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="HeadRouter: FLUX RF-Inversion editing")
    parser.add_argument(
        "--model",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="FLUX model id or local path",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Input image path (png/jpg) or cached .pt latents path",
    )
    parser.add_argument(
        "--prompt_list",
        nargs="+",
        required=True,
        help="Two prompts: source description and target edit description",
    )
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--strength", type=float, default=0.95)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--gamma", type=float, default=0.5, help="RF-Inversion gamma")
    parser.add_argument("--eta", type=float, default=0.9, help="RF-Inversion eta")
    parser.add_argument("--start_timestep", type=int, default=0)
    parser.add_argument("--stop_timestep", type=int, default=8)
    parser.add_argument("--seed", type=int, default=8889)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(ROOT / "outputs"),
        help="Directory to save results",
    )
    parser.add_argument(
        "--block_id",
        type=int,
        nargs="+",
        default=[0],
        help="Optional head indices (legacy flag; HeadRouter uses all heads)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.prompt_list) < 2:
        raise ValueError("--prompt_list needs at least two prompts: [source, target]")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = FluxRFInversionPipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(device)

    image_path = str(Path(args.image).expanduser().resolve())
    file_name = Path(image_path).stem

    torch.manual_seed(args.seed)
    generator = torch.Generator("cpu").manual_seed(args.seed)

    images = pipe(
        prompt=args.prompt_list,
        image=image_path,
        num_inference_steps=args.num_inference_steps,
        strength=args.strength,
        generator=generator,
        guidance_scale=args.guidance_scale,
        gamma=args.gamma,
        eta=args.eta,
        start_timestep=args.start_timestep,
        stop_timestep=args.stop_timestep,
        block_id=list(args.block_id),
    ).images

    out_dir = Path(args.output_dir) / file_name
    out_dir.mkdir(parents=True, exist_ok=True)
    joined = np.concatenate(images, axis=1)
    out_name = (
        f"gamma{args.gamma}_eta{args.eta}"
        f"_s{args.start_timestep}e{args.stop_timestep}_{args.seed}.png"
    )
    out_path = out_dir / out_name
    Image.fromarray(joined).save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
