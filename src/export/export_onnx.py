import torch
from pathlib import Path

from src.models.generator import Generator

def main():
    model_path = Path("models/gan_generator.pt")
    out_path = Path("models/gan_generator.onnx")

    assert model_path.exists(), "gan_generator.pt not found"

    # Load model
    model = Generator()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Dummy input: 4-channel LR (RGB + NIR)
    dummy_input = torch.randn(1, 4, 256, 256)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        out_path,
        opset_version=18,
        input_names=["lr_image"],
        output_names=["sr_image"],
        dynamic_axes={
            "lr_image": {0: "batch", 2: "height", 3: "width"},
            "sr_image": {0: "batch", 2: "height", 3: "width"},
        },
        export_params=True,
        do_constant_folding=True,
        verbose=False,
        dynamo=False
    )

    print(f"✅ ONNX export successful → {out_path}")

if __name__ == "__main__":
    main()
