import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Union


# color list
_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.857, 0.857, 0.857,
        1.000, 1.000, 1.000
    ]
).astype(np.float32).reshape(-1, 3)


class BBoxVisualizer:
    """Visualizer for Bounding Boxes.

    Attributes:
        classes: A list of class names.
        colors: A list of colors; each color is a tuple of 3 intengers.
    """
    def __init__(self, classes: List[str]):
        """Initilize the BBoxVisualizer.

        Args:
            classess: A list of class names; each class name is a string.
        """
        self.classes = classes
        self.colors = self._generate_unique_colors(len(classes))
    
    def _generate_unique_colors(self,
                                num_colors: int) -> List[Tuple[int,int,int]]:
        """Generate a list of unique colors.

        Args:
            num_colors: Number of colors to be generated.

        Returns:
            colors: A list; each color element is a tuple of 3 integers.
        """
        if num_colors <= len(_COLORS):
            colors = [tuple((c*255).astype(np.uint8)) for c in random.sample(list(_COLORS), num_colors)]
        else:
            colors = set(tuple(c) for c in (_COLORS*255).astype(np.uint8))
            while len(colors) < num_colors:
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                colors.add((r, g, b))
            colors = list(colors)
            
        return colors
    
    def visualize_bbox(self,
                       img: np.ndarray,
                       bboxes: List[List[int]],
                       labels: List[int],
                       scores: Union[List[float], None]=None,
                       top: bool=False) -> np.ndarray:
        """Visualize multiple bounding boxes with labels and scores(if they are exist) for an image.

        Args:
            img: The image which need to be added bounding boxes and labels. Shape of `(C,H,W)`.
            bboxes: Bounding boxes. The format of a box is `(xmin, ymin, xmax, ymax)`.
            labels: Labels associated with classes. Each label is a label index.
            scores: Confidence or objectiveness scores.
            top: Whether to place text labels on the top of bounding boxes.

        Returns:
            img: An image in `numpy.ndarray`.
        """
        if scores is None:
            assert len(bboxes) == len(labels), "bboxes must have same length with labels"
            for label_idx, bbox in zip(labels, bboxes):
                label_idx = label_idx % len(self.classes)
                label = self.classes[label_idx]
                color = self.colors[label_idx]
                img = draw_rectangle(img, bbox, bbox_color=color, thickness=1)
                img = add_label(img, label, bbox, size=12, draw_bg=True,
                                text_bg_color=color, top=top)
        else:
            assert len(bboxes) == len(labels) == len(scores),\
                   "bboxes, labels, and scores must have same length"
            for label_idx, bbox, score in zip(labels, bboxes, scores):
                label_idx = label_idx % len(self.classes)
                label = "{} {:.3f}".format(self.classes[label_idx], score)
                color = self.colors[label_idx]
                img = draw_rectangle(img, bbox, bbox_color=color, thickness=1)
                img = add_label(img, label, bbox, size=12, draw_bg=True,
                                text_bg_color=color, top=top)
        
        return img


def draw_rectangle(img: np.ndarray,
                   bbox: List[int],
                   bbox_color: Tuple[int]=(255, 255, 255),
                   thickness: int=1,
                   is_opaque: bool=False,
                   alpha: float=0.5):
    """Draw the rectangle(bounding box) on the given image.

    Args:
        img: The image which need to be added a bounding box.
        bbox: Bounding box. The format is `(xmin, ymin, xmax, ymax)`.
        bbox_color: Color of the box.
        thickness: The width of the outline of the box.
        is_opaque: Whether to make the box opaque. If the box is opaque,
                   the box will be filled with `bbox_color`.
        alpha: The opacity of the box if `is_opaque` is `True`.

    Returns:
        output_img: A result image in `numpy.ndarray`.
    """
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img, mode='RGBA')
    if not is_opaque:
        draw.rectangle(tuple(bbox), outline=bbox_color, width=thickness)
    else:
        fill_color = bbox_color+(int(alpha*255),)
        draw.rectangle(tuple(bbox), fill=fill_color)
    output_img = np.asarray(pil_img, dtype=np.uint8)

    return output_img


def add_label(img: np.ndarray,
              label: str,
              bbox: List[int],
              size: int=10,
              draw_bg: bool=True,
              text_bg_color: Tuple[int]=(255, 255, 255),
              alpha: float=0.5,
              text_color: Tuple[int]=(0, 0, 0),
              top: bool=True,
              font_fp: str=None) -> np.ndarray:
    """Add label to the image.

    Args:
        img: The image which need to be added a text label.
        label: The text label which is a `str`.
        bbox: Bounding box. The format is `(xmin, ymin, xmax, ymax)`.
        size: Font size of the text label.
        draw_bg: Whether to draw background for the text label.
        text_bg_color: Color of the background of the text label.
        alpha: The opacity of the background of the text label.
        text_color: Color of the text label.
        top: Whether to place the text label on the top of the bounding box.
        font_fp: File path to the font file.

    Returns:
        output_img: A result image in `numpy.ndarray`.
    """
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img, mode='RGBA')
    if font_fp is not None:
        font = ImageFont.truetype(font_fp, size=size)
    else:
        font = None
    label_box = font.getbbox(label)
    label_width = label_box[2] - label_box[0]
    label_height = label_box[3] - label_box[1]
    if top:
        if draw_bg:
            fill_color = text_bg_color+(int(alpha*255),)
            draw.rectangle((bbox[0], bbox[1], bbox[0]+label_width, bbox[1]-label_height), fill=fill_color)
        draw.text((bbox[0], bbox[1]), label, fill=text_color, font=font, anchor='lb')
    else:
        if draw_bg:
            fill_color = text_bg_color+(int(alpha*255),)
            draw.rectangle((bbox[0], bbox[1], bbox[0]+label_width, bbox[1]+label_height), fill=fill_color)
        draw.text((bbox[0], bbox[1]), label, fill=text_color, font=font, anchor='lt')
    
    output_img = np.asarray(pil_img, dtype=np.uint8)

    return output_img

def draw_multiple_rectangles(img: np.ndarray,
                             bboxes: List[List[int]],
                             bbox_color: Tuple[int]=(255, 255, 255),
                             thickness: int=3,
                             is_opaque: bool=False,
                             alpha: float=0.5) -> np.ndarray:
    """Draw multiple rectangles(boxes) to the given image.

    Args:
        img: The image which need to be added bounding boxes.
        bboxes:  Bounding boxes. The format of a box is `(xmin, ymin, xmax, ymax)`.
        bbox_color: Color of the box.
        thickness: The width of the outline of the box.
        is_opaque: Whether to make the box opaque. If the box is opaque,
                   the box will be filled with `bbox_color`.
        alpha: The opacity of the box if `is_opaque` is `True`.

    Returns:
        img: A result image in `numpy.ndarray`.
    """
    for bbox in bboxes:
        img = draw_rectangle(img, bbox, bbox_color, thickness, is_opaque, alpha)
    
    return img

def add_multiple_labels(img: np.ndarray,
                        labels: List[str],
                        bboxes: List[List[int]],
                        size: int=10,
                        draw_bg: bool=True,
                        text_bg_color: Tuple[int]=(255, 255, 255),
                        alpha: float=0.5,
                        text_color: Tuple[int]=(0, 0, 0),
                        top: bool=True,
                        font_fp: str=NotImplementedError) -> np.ndarray:
    """Add multiple labels to the given image.

    Each text label corresponds to a specific bounding box.

    Args:
        img: The image which need to be added text labels.
        labels: The text labels which is a list of `str`.
        bboxes: Bounding boxes. The format of a box is `(xmin, ymin, xmax, ymax)`.
        size: Font size of the text label.
        draw_bg: Whether to draw background for the text label.
        text_bg_color: Color of the background of the text label.
        alpha: The opacity of the background of the text label.
        text_color: Color of the text label.
        top: Whether to place the text label on the top of the bounding box.
        font_fp: File path to the font file.

    Returns:
        img: A result image in `numpy.ndarray`.
    """
    for label,bbox in zip(labels, bboxes):
        img = add_label(img, label, bbox, size, draw_bg, text_bg_color,
                        alpha, text_color, top, font_fp)
        
    return img