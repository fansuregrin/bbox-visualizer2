"""Top-level package for bbox-visualizer2."""

__author__ = 'Fansure Grin'
__email__ = 'pwz113436@gmail.com'
__version__ = '0.0.2'

from .bbox_visualizer2 import (
    draw_rectangle,
    add_label,
    draw_multiple_rectangles,
    add_multiple_labels,
    BBoxVisualizer
)

__all__ = [
    draw_rectangle,
    add_label,
    draw_multiple_rectangles,
    add_multiple_labels,
    BBoxVisualizer
]