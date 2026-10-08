"""Check dependencies and perform a tiny forward/backward step on the chosen device."""

import json
import platform
import sys
from pathlib import Path

import matplotlib
import numpy
import pandas
import scipy
import torch
import torchvision


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs" / "setup"
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 8, 3, padding=1), torch.nn.SiLU(),
        torch.nn.Conv2d(8, 1, 3, padding=1),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(8, 1, 28, 28, device=device)
    target = torch.randn_like(x)
    prediction = model(x)
    assert prediction.shape == x.shape
    loss = (prediction - target).square().mean()
    loss.backward()
    for parameter in model.parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError("Missing or non-finite gradient")
    optimizer.step()
    if device.type == "mps":
        torch.mps.synchronize()
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise RuntimeError("Non-finite parameters after optimizer step")

    report = {
        "python": platform.python_version(), "executable": sys.executable,
        "platform": platform.platform(), "architecture": platform.machine(),
        "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                     "numpy": numpy.__version__, "scipy": scipy.__version__,
                     "matplotlib": matplotlib.__version__, "pandas": pandas.__version__},
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(), "device_tested": str(device),
        "smoke_check": "finite convolution forward/backward and Adam optimizer step",
        "batch_shape": list(x.shape), "status": "passed",
    }
    (output / "environment_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
