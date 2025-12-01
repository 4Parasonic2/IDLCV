import os
import glob
import random
from PIL import Image
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Script to visualize data
path = "/dtu/datasets1/02516/potholes/"

img_path = path + "images/"
ann_path = path + "annotations/"
res_path = "/zhome/4d/5/147570/IDLCV/Project_4/results"



# patch plt.show so each generated figure is saved to results_dir using the image basename from the title
_orig_show = plt.show
def _save_show(*args, **kwargs):
    fig = plt.gcf()
    fname = 'figure'
    try:
        if fig.axes:
            title = fig.axes[0].get_title()
            fname = title.split('  |  ')[0].strip() or fname
    except Exception:
        pass
    safe = "".join(c for c in fname if c.isalnum() or c in (' ', '.', '_', '-')).rstrip()
    out = os.path.join(res_path, safe + '.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
plt.show = _save_show


# collect image files 
image_files = glob.glob(os.path.join(img_path, '*.png'))
if not image_files:
    raise RuntimeError("No images found in img_path")

# sample up to 6 examples
sample_files = random.sample(image_files, min(6, len(image_files)))

for img_file in sample_files:
    base = os.path.splitext(os.path.basename(img_file))[0]
    ann_file = os.path.join(ann_path, base + '.xml')

    img = Image.open(img_file).convert('RGB')
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img)
    ax.axis('off')

    title = os.path.basename(img_file)
    if os.path.exists(ann_file):
        try:
            tree = ET.parse(ann_file)
            root = tree.getroot()
            objects = root.findall('object')
            labels = []
            for obj in objects:
                name_el = obj.find('name')
                name = name_el.text if name_el is not None else 'object'
                labels.append(name)
                bnd = obj.find('bndbox')
                if bnd is None:
                    continue
                xmin = int(float(bnd.find('xmin').text))
                ymin = int(float(bnd.find('ymin').text))
                xmax = int(float(bnd.find('xmax').text))
                ymax = int(float(bnd.find('ymax').text))
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                         linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(xmin, max(ymin - 6, 0), name, color='yellow',
                        fontsize=10, backgroundcolor='black')
            if labels:
                title += "  |  " + ", ".join(labels)
        except Exception as e:
            title += f"  |  annotation parse error: {e}"
    else:
        title += "  |  no annotation"

    ax.set_title(title)
    plt.show()

