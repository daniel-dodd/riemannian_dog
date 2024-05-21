"""Utilities for working with plots. """
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))

__all__ = [
    "set_style_and_font",
    "get_cols",
]


def set_style_and_font():
    font = mpl.font_manager.FontEntry(
        fname=os.path.join(dir_path, "OggRegular.ttf"), name="my_font"
    )
    mpl.font_manager.fontManager.ttflist.append(font)
    mpl.rcParams.update(
        {
            "font.family": font.name,
        }
    )
    plt.style.use(os.path.join(dir_path, "plotstyle.mplstyle"))


def get_cols():
    return plt.rcParams["axes.prop_cycle"].by_key()["color"]
