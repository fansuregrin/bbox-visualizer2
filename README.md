# BBox-Visualizer2

This is a python package for visualizing bounding boxes and adding labels to a given image. The creation of this package is inspired by [shoumikchow/bbox-visualizer](https://github.com/shoumikchow/bbox-visualizer), but I re-implement it through `Pillow` instead of `cv2`.

The format of a bounding box is `(xmin, ymin, xmax, ymax)`.

Documentation: [https://fansuregrin.github.io/bbox-visualizer2](https://fansuregrin.github.io/bbox-visualizer2/)

## Installation
### From `Pypi`
```
pip install bbox-visualizer2
```

### From source
```
git clone https://github.com/fansuregrin/bbox-visualizer2.git bbv2
cd bbv2
python3 -m pip install --upgrade build
python3 -m build
pip install dist/bbox_visualizer2-0.0.2-py3-none-any.whl
```

## Usage
You can see [demos](./demos/) for details.

### Multiple bboxes visualization
<details><summary>Code examples</summary>
<p>

```python
classes = [
    'person', 'bird',
    'cat', 'cow',
    'dog', 'horse',
    'sheep', 'aeroplane',
    'bicycle', 'boat',
    'bus', 'car',
    'motorbike', 'train',
    'bottle', 'chair',
    'diningtable', 'pottedplant',
    'sofa', 'tvmonitor',
]
font_path = 'assets/fonts/LXGWWenKai-Regular.ttf'
img_path = 'assets/images/000623.jpg'
pil_img = Image.open(img_path)
img = np.asarray(pil_img, dtype=np.uint8)
xml_filepath = 'assets/annotations/000623.xml'
boxes, labels = get_annot_info(xml_filepath, classes)

bbox_visualizer = bbv.BBoxVisualizer(classes, font_path)
img = bbox_visualizer.visualize_bbox(img, boxes, labels)
```

</p>
</details>

![](assets/images/000623_result.png)

![](assets/images/000127_result.png)

![](assets/images/000014_result.png)

### Single bbox visualization

#### (1) Label on the top of the bbox
<details><summary>Code examples</summary>
<p>

```python
label = 'cow'
box = [299, 160, 446, 252]
color = (255, 255, 0)
text_color = (0, 0, 0)

img1 = bbv.draw_rectangle(img, box, bbox_color=color, thickness=1)
img1 = bbv.add_label(img1, label, box, size=12, draw_bg=True,
                     text_bg_color=color, alpha=0.5, text_color=text_color,
                     top=True, font_fp=font_path)
```

</p>
</details>

![](assets/images/000013_result1.png)

#### (2) Label inside the box
<details><summary>Code examples</summary>
<p>

```python
label = 'cow'
box = [299, 160, 446, 252]
color = (255, 255, 0)
text_color = (0, 0, 0)

img2 = bbv.draw_rectangle(img, box, bbox_color=color, thickness=3)
img2 = bbv.add_label(img2, label, box, size=12, draw_bg=True,
                     text_bg_color=color, alpha=0.5, text_color=text_color,
                     top=False, font_fp=font_path)
```

</p>
</details>

![](assets/images/000013_result2.png)

#### (3) Set the box to opaque
<details><summary>Code examples</summary>
<p>

```python
label = 'cow'
box = [299, 160, 446, 252]
color = (255, 255, 0)
text_color = (0, 0, 0)

img3 = bbv.draw_rectangle(img, box, bbox_color=color, is_opaque=True, alpha=0.5)
img3 = bbv.add_label(img3, label, box, size=12, draw_bg=True,
                     text_bg_color=color, alpha=0.5, text_color=text_color,
                     top=False, font_fp=font_path)
```

</p>
</details>

![](assets/images/000013_result3.png)