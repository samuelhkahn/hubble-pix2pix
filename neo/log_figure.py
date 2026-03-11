"""Utilities for logging astronomical image figures to Comet ML."""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.visualization import simple_norm


def log_figure(img, fig_name, experiment, cmap="plasma", set_lims=False,
               lims=(-1, 1), stretch="linear"):
    """Create and log an astronomical image figure to a Comet ML experiment.

    Renders the image with a colorbar using astronomical conventions
    (origin="lower") and logs it to the experiment tracker.

    Args:
        img: 2D numpy array of pixel values.
        fig_name: Name/title for the logged figure.
        experiment: Comet ML Experiment instance.
        cmap: Matplotlib colormap name.
        set_lims: If True, use explicit vmin/vmax from ``lims``.
        lims: Tuple of (vmin, vmax) for color scaling.
        stretch: Astropy normalization stretch (e.g., "linear", "log", "asinh").
    """
    f, ax = plt.subplots()

    if not set_lims:
        im = ax.imshow(img, cmap=cmap, origin="lower",
                        norm=simple_norm(img, stretch=stretch))
    else:
        im = ax.imshow(img, cmap=cmap, origin="lower",
                        vmin=lims[0], vmax=lims[1])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(im, cax=cax, orientation='vertical')

    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    experiment.log_figure(figure_name=fig_name, figure=f)
    plt.close()
