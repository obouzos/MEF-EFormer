# from argparse import ArgumentParser
# from pathlib import Path

# import torch
# import torchvision.transforms.functional as TF
# from PIL import Image

# from MEF_EFormer import MEF_EFormer


# ROOT_DIR = Path(__file__).resolve().parent
# DEFAULT_WEIGHTS = ROOT_DIR / "MEF_EFormer.pth"


# def load_image(path: Path) -> torch.Tensor:
#     with Image.open(path) as image:
#         rgb_image = image.convert("RGB")
#         image_tensor = TF.to_tensor(rgb_image)

#     return image_tensor.unsqueeze(0)

# def save_image(tensor: torch.Tensor,path: Path,) -> None:
#     image_tensor = tensor.detach().cpu()

#     if image_tensor.ndim == 4:
#         image_tensor = image_tensor.squeeze(0)

#     output_path = path.expanduser().resolve()
#     output_path.parent.mkdir(parents=True, exist_ok=True)

#     print(f"Saving to: {output_path!r}")

#     output_image = TF.to_pil_image(image_tensor)

#     with output_path.open("wb") as output_file:
#         output_image.save(output_file,format="PNG")


# def load_model(weights_path: Path,device: torch.device) -> MEF_EFormer:
#     weights = torch.load(weights_path,map_location="cpu",weights_only=True)

#     model = MEF_EFormer() 
#     model.load_state_dict(weights,strict=True)

#     model = model.to(device)
#     model.eval()

#     return model


# def main() -> None:
#     parser = ArgumentParser(
#         description="MEF-EFormer multi-exposure image fusion inference."
#     )

#     parser.add_argument(
#         "--under",
#         type=Path,
#         required=True,
#         help="Path to the underexposed RGB image.",
#     )

#     parser.add_argument(
#         "--over",
#         type=Path,
#         required=True,
#         help="Path to the overexposed RGB image.",
#     )

#     parser.add_argument(
#         "--output",
#         type=Path,
#         default=ROOT_DIR / "fused.png",
#         help="Path for the fused PNG output.",
#     )

#     parser.add_argument(
#         "--weights",
#         type=Path,
#         default=DEFAULT_WEIGHTS,
#         help="Path to the MEF_EFormer.pth weights file.",
#     )

#     args = parser.parse_args()


#     device = torch.device(
#         "cuda" if torch.cuda.is_available() else "cpu"
#     )

#     under = load_image(args.under)
#     over = load_image(args.over)

#     model = load_model(weights_path=args.weights,device=device)

#     under = under.to(device=device,dtype=torch.float32)
#     over = over.to(device=device,dtype=torch.float32)
#     output = model.inference(under,over)

#     save_image(tensor=output,path=args.output)

#     print(f"Device: {device}")
#     print(f"Underexposed image: {args.under}")
#     print(f"Overexposed image: {args.over}")
#     print(f"Input shape: {tuple(under.shape)}")
#     print(f"Fused image saved to: {args.output}")


# if __name__ == "__main__":
#     main()



from argparse import ArgumentParser
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from MEF_EFormer import MEF_EFormer


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT_DIR / "MEF_EFormer.pth"


def load_image(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image_tensor = TF.to_tensor(image.convert("RGB"))

    return image_tensor.unsqueeze(0)


def save_image(tensor: torch.Tensor, path: Path) -> None:
    image_tensor = tensor.detach().float().cpu()

    if image_tensor.ndim == 4:
        image_tensor = image_tensor.squeeze(0)

    output_path = path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_image = TF.to_pil_image(image_tensor)

    with output_path.open("wb") as output_file:
        output_image.save(output_file, format="PNG")


def load_model(weights_path: Path, device: torch.device, dtype: torch.dtype) -> MEF_EFormer:
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)

    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model = MEF_EFormer()
    model.load_state_dict(checkpoint, strict=True)

    return model.to(device=device, dtype=dtype).eval()


def main() -> None:
    parser = ArgumentParser(description="MEF-EFormer multi-exposure image fusion inference.")
    parser.add_argument("--under", type=Path, required=True, help="Path to the underexposed RGB image.")
    parser.add_argument("--over", type=Path, required=True, help="Path to the overexposed RGB image.")
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "fused.png", help="Path for the fused PNG output.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Path to the MEF_EFormer.pth weights file.")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32", help="Native inference precision. FP16 requires CUDA.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.precision == "fp16" and device.type != "cuda":
        raise RuntimeError("Native FP16 inference requires a CUDA device.")

    dtype = torch.float16 if args.precision == "fp16" else torch.float32

    under = load_image(args.under)
    over = load_image(args.over)

    if under.shape != over.shape:
        raise ValueError(f"Input images must have the same shape, but received {tuple(under.shape)} and {tuple(over.shape)}.")

    model = load_model(weights_path=args.weights, device=device, dtype=dtype)
    under = under.to(device=device, dtype=dtype)
    over = over.to(device=device, dtype=dtype)

    with torch.inference_mode():
        output = model(under, over).clamp_(0.0, 1.0)

    if not bool(torch.isfinite(output).all().item()):
        raise RuntimeError(f"Model produced non-finite values with {args.precision} inference.")

    output_dtype = output.dtype
    save_image(output, args.output)

    print(f"Device: {device}")
    print(f"Precision: {args.precision}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"Input dtype: {under.dtype}")
    print(f"Output dtype: {output_dtype}")
    print(f"Input shape: {tuple(under.shape)}")
    print(f"Underexposed image: {args.under}")
    print(f"Overexposed image: {args.over}")
    print(f"Fused image saved to: {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()