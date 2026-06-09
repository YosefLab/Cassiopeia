"""Top level for plotting."""

from .deprecated import (
    Tree3D,
    labels_from_coordinates,
    plot_matplotlib,
    plot_plotly,
    upload_and_export_itol,
)

__all__ = [
    "upload_and_export_itol",
    "plot_matplotlib",
    "plot_plotly",
    "labels_from_coordinates",
    "Tree3D",
]
