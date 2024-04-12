import json
import os

vg_captions = json.load(open("/home/cap6412.student3/datasets/vg/annotations/vg_caption.json", "r"))

VG_100K_dict = {}
VG_100K_2_dict = {}

for img in os.listdir('/datasets/VG/VG_100K'):
    VG_100K_dict[img] = True

for img in os.listdir('/datasets/VG/VG_100K_2'):
    VG_100K_2_dict[img] = True

for i, caption in enumerate(vg_captions):
    image = caption["image"].split("/")[-1]
    
    if image in VG_100K_dict:
        image = "VG_100K/" + image
    elif image in VG_100K_2_dict:
        image = "VG_100K_2/" + image
    else:
        raise ValueError(f"image: {image} not found in VG_100K or VG_100K_2")
    
    vg_captions[i]["image"] = image


with open("/home/cap6412.student3/datasets/vg/annotations/vg_caption.json", "w") as f:
    json.dump(vg_captions, f)

