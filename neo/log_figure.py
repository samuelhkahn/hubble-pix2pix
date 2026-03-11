"""Utilities for logging astronomical image figures to Comet ML.

Provides a single function that renders a 2D pixel array as a matplotlib
figure (with colorbar, astronomical orientation, and optional normalization)
and logs it to a Comet ML experiment for visual inspection during training.
"""

import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def log_figure(
    img,
    fig_name: str,
    experiment,
    cmap: str = "plasma",
    set_lims: bool = False,
    lims: tuple = (-1, 1),
    stretch: str = "linear",
) -> None:
    """Create and log an astronomical image figure to Comet ML.

    The image is displayed with ``origin="lower"`` (standard astronomical
    convention where the origin is at the bottom-left) and a colorbar for
    quantitative pixel-value reference.

    Args:
        img: 2D numpy array of pixel values to display.
        fig_name: Name/title under which the figure appears in Comet ML.
        experiment: ``comet_ml.Experiment`` instance.
        cmap: Matplotlib colormap name (e.g., ``"plasma"``, ``"gray"``).
        set_lims: If ``True``, use explicit ``vmin``/``vmax`` from ``lims``
            instead of auto-scaling.
        lims: ``(vmin, vmax)`` tuple for color scaling (only used when
            ``set_lims=True``).  Useful for symmetric difference maps.
        stretch: Astropy ``simple_norm`` stretch type (e.g., ``"linear"``,
            ``"log"``, ``"asinh"``).  Only used when ``set_lims=False``.
    """
    f, ax = plt.subplots()

    if not set_lims:
        # Use astropy's simple_norm for automatic stretch normalization.
        im = ax.imshow(
            img,
            cmap=cmap,
            origin="lower",
            norm=simple_norm(img, stretch=stretch),
        )
    else:
        # Use explicit vmin/vmax (e.g., for symmetric residual maps).
        im = ax.imshow(
            img, cmap=cmap, origin="lower", vmin=lims[0], vmax=lims[1]
        )

    # Add a colorbar to the right side of the plot.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    f.colorbar(im, cax=cax, orientation="vertical")

    # Remove axis ticks (pixel coordinates are not meaningful for display).
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    experiment.log_figure(figure_name=fig_name, figure=f)
    plt.close()
