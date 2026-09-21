"""Shared setup and palette for the Hyperparameters lecture figures.

The figures for the three blocks are drawn by `figures_intro.py`,
`figures_search.py` and `figures_budget.py`; they only take the colors and
the save function from here, to keep a consistent look. The actual numbers
of the lecture are computed by `optuna_hyperparameters.py`.

The palette is shared across the whole course.
"""

import io
import os
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams["svg.hashsalt"] = "aui-hyperparameters"
SVG_METADATA = {"Date": None}

BLUE = "#12355b"
TEAL = "#007f86"
RED = "#c73e1d"
ORANGE = "#e9a23b"
LIGHT_BLUE = "#d9eef7"
GREY = "#eef2f4"


def save_figure(fig, name: str) -> None:
    """Saves the figure as SVG; with the AUI_PREVIEWS variable also as a PNG to check."""
    fig.savefig(OUT / f"{name}.svg", metadata=SVG_METADATA, bbox_inches="tight")
    previews = os.environ.get("AUI_PREVIEWS")
    if previews:
        directory = Path(previews)
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / f"{name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_gif(frames, durations, name: str, dpi: int = 120) -> None:
    """Saves a sequence of figures as a looping GIF.

    `frames` are the finished figures in playback order, `durations` their
    display times in milliseconds. **The first frame also doubles as the
    static fallback** (in a PDF or in print only it shows), so it must
    carry the final state.
    """
    images = []
    for fig in frames:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=dpi)
        plt.close(fig)
        buffer.seek(0)
        images.append(Image.open(buffer).convert("RGB")
                      .quantize(colors=96, method=Image.MEDIANCUT))
    images[0].save(OUT / f"{name}.gif", save_all=True, append_images=images[1:],
                   duration=list(durations), loop=0, optimize=True, disposal=2)
