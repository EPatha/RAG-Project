"""Optional text-to-image example.

Install separately when needed:
    pip install diffusers transformers torch
"""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--out", default="generated.png")
    args = parser.parse_args()

    try:
        import torch
        from diffusers import StableDiffusionPipeline
    except ImportError as exc:
        raise SystemExit(
            "Install optional diffusion dependencies first: "
            "pip install diffusers transformers torch"
        ) from exc

    pipe = StableDiffusionPipeline.from_pretrained(args.model)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    pipe(args.prompt).images[0].save(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
