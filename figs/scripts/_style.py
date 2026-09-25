"""Shared matplotlib style for paper figures.

All paper plots go through
`apply_rc()` here, which composes scienceplots `[science, bright,
high-contrast, no-latex]` with the Tol `bright_extended` color cycler and a few
paper-specific overrides (serif body font, type-42
embedded fonts for editable PDFs).

The explicit semantic color maps below (`STANCE_COLOR`,
`DIALECT_COLOR`) override the cycler for axes where a specific
variable -> color binding matters. Use them by name (e.g. `STANCE_COLOR['left']`)
rather than relying on cycler order.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

try:
    import scienceplots  # noqa: F401  registers science/bright/high-contrast
except ImportError as e:
    raise ImportError(
        "scienceplots is required for the paper figure style — "
        "install with `pip install scienceplots`."
    ) from e


# Tol-derived bright_extended — default cycler for any plot that doesn't
# bind a semantic color explicitly. 11 distinct hues (#E60000 appears twice).
BRIGHT_EXTENDED = [
    "#4477AA",  # blue
    "#E60000",  # red
    "#228833",  # green
    "#CCBB44",  # yellow
    "#66CCEE",  # light blue
    "#AA3377",  # purple
    "#BBBBBB",  # gray
    "#000000",  # black
    "#EE7733",  # orange
    "#33BBEE",  # cyan
    "#E60000",  # dark red
    "#0077BB",  # dark blue
]


ENCODER_ORDER = [
    "bge-large-en-v1.5",
    "qwen3-embedding-8b",
    "llama-embed-nemotron-8b",
    "octen-embedding-8b",
    "text-embedding-3-large",
]

ENCODER_DISPLAY = {
    "bge-large-en-v1.5":        "BGE",
    "qwen3-embedding-8b":       "Qwen",
    "llama-embed-nemotron-8b":  "Nemotron",
    "octen-embedding-8b":       "Octen",
    "text-embedding-3-large":   "OpenAI",
}

STANCE_COLOR = {
    "left":    "#4477AA",  # blue
    "neutral": "#BBBBBB",  # gray
    "right":   "#E60000",  # dark red
}

# Dialect colors deliberately kept off the political blue/red pair so
# AAL/WME figures don't read as "political left/right" at a glance.
DIALECT_COLOR = {
    "wme":   "#EE7733",  # orange
    "aal":   "#AA3377",  # purple
}


def apply_rc() -> None:
    """Apply the project-standard paper style.

    Order matters: scienceplots styles first (which set base rcParams),
    then the bright_extended cycler + paper-specific overrides on top.
    """
    plt.style.use(["science", "bright", "high-contrast", "no-latex"])
    mpl.rcParams.update({
        "axes.prop_cycle":    cycler(color=BRIGHT_EXTENDED),
        # Use bundled fonts and mathtext: rendering needs no system TeX/fonts.
        "text.usetex":        False,
        "font.family":        "serif",
        "font.serif":         ["DejaVu Serif"],
        "mathtext.fontset":   "dejavuserif",
        "font.size":          9,
        "axes.titlesize":     9.5,
        "axes.labelsize":     9,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "legend.fontsize":    8,
        "legend.frameon":     False,
        "savefig.bbox":       "tight",
        "savefig.dpi":        300,
        "pdf.fonttype":       42,  # editable text in PDFs
        "ps.fonttype":        42,
    })


def save_both(fig, path_no_ext: str) -> None:
    from pathlib import Path

    Path(path_no_ext).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{path_no_ext}.pdf")
    fig.savefig(f"{path_no_ext}.png", dpi=300)
    plt.close(fig)
