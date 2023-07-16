import bbox_visualizer2 as bbv
import xmltodict
import os
import numpy as np
from PIL import Image


def get_annot_info(xml_filepath):
    with open(xml_filepath, 'r') as xml_f:
        annotation = xmltodict.parse(xml_f.read(), encoding='utf8', )['annotation']
    
    objects = annotation['object']
    if not isinstance(objects, list):
        objects = [objects]
    labels = []
    boxes = []

    for obj in objects:
        labels.append(obj['name'])
        bnd_box = obj['bndbox']
        xmin = float(bnd_box['xmin'])
        ymin = float(bnd_box['ymin'])
        xmax = float(bnd_box['xmax'])
        ymax = float(bnd_box['ymax'])
        boxes.append([xmin, ymin, xmax, ymax])

    return boxes, labels


if __name__ == '__main__':
    font_path = 'assets/fonts/LXGWWenKai-Regular.ttf'
    img_path = 'assets/images/000623.jpg'
    pil_img = Image.open(img_path)
    img = np.asarray(pil_img, dtype=np.uint8)
    xml_filepath = 'assets/annotations/000623.xml'
    boxes, labels = get_annot_info(xml_filepath)

    img = bbv.draw_multiple_rectangles(img, boxes)
    img = bbv.add_multiple_labels(img, labels, boxes, font_fp=font_path)
    result_img = Image.fromarray(img)
    save_name = '.'.join(os.path.basename(img_path).split('.')[:-1]) + '_result2'
    result_img.save(f'assets/images/{save_name}.png')