import bbox_visualizer2 as bbv
import numpy as np
import os
from PIL import Image


font_path = 'assets/fonts/LXGWWenKai-Regular.ttf'
img_path = 'assets/images/000013.jpg'
pil_img = Image.open(img_path)
img = np.asarray(pil_img, dtype=np.uint8)

label = 'cow'
box = [299, 160, 446, 252]
color = (255, 255, 0)
text_color = (0, 0, 0)

img1 = bbv.draw_rectangle(img, box, bbox_color=color, thickness=1)
img1 = bbv.add_label(img1, label, box, size=12, draw_bg=True,
                     text_bg_color=color, alpha=0.5, text_color=text_color,
                     top=True, font_fp=font_path)
pil_img1 = Image.fromarray(img1)
save_name = '.'.join(os.path.basename(img_path).split('.')[:-1]) + '_result1'
pil_img1.save(f'assets/images/{save_name}.png')

img2 = bbv.draw_rectangle(img, box, bbox_color=color, thickness=3)
img2 = bbv.add_label(img2, label, box, size=12, draw_bg=True,
                     text_bg_color=color, alpha=0.5, text_color=text_color,
                     top=False, font_fp=font_path)
pil_img2 = Image.fromarray(img2)
save_name = '.'.join(os.path.basename(img_path).split('.')[:-1]) + '_result2'
pil_img2.save(f'assets/images/{save_name}.png')

img3 = bbv.draw_rectangle(img, box, bbox_color=color, is_opaque=True, alpha=0.5)
img3 = bbv.add_label(img3, label, box, size=12, draw_bg=True,
                     text_bg_color=color, alpha=0.5, text_color=text_color,
                     top=False, font_fp=font_path)
pil_img3 = Image.fromarray(img3)
save_name = '.'.join(os.path.basename(img_path).split('.')[:-1]) + '_result3'
pil_img3.save(f'assets/images/{save_name}.png')