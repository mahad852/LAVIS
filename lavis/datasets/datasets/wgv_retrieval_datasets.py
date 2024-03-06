import os
from collections import OrderedDict

from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]
        visual_key = "image" if "image" in ann else "video"

        return OrderedDict(
            {
                "file": ann[visual_key],
                "caption": ann["caption"],
                visual_key: sample[visual_key],
            }
        )
    
def clean_location_text(text: str) -> str:
    return ' '.join(text.split('_'))

class WGVRetrievalEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """    
        self.vis_root = vis_root
        self.city_info = {}

        countries = set()

        with open(ann_paths[0], 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                info = line[:-1].split(',')
                self.city_info[info[0]] = {'state' : info[1], 'country' : info[2], 'continent': info[3]}
                countries.add(info[2])

        txt_id = 0
        self.text = []
        self.txt2img = {}

        self.country_texts = {}
        self.country_txt_ids = {} 

        for country in countries:
            country_texts = [f'a photo i took in {clean_location_text(country)}.',
                             f'a photo showing the country of {clean_location_text(country)}.',
                             f'a photo from my home country of {clean_location_text(country)}.']
            
            self.country_texts[country] = country_texts
            self.country_txt_ids[country] = []

            for country_text in country_texts:
                self.text.append(country_text)
                self.txt2img[txt_id] = []
                self.country_txt_ids[country].append(txt_id)
                txt_id += 1
                
        self.annotation = []

        self.image = []
        self.img2txt = {}

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        image_files = os.listdir(self.vis_root)[:8000]

        for i, im_name in enumerate(image_files):
            city = '_'.join(im_name.split('_')[0:-2])
            country = self.city_info[city]['country']

            self.annotation.append({'image' : im_name, 'instance_id' : str(i), 'caption' : self.country_texts[country]})
            

            self.img2txt[i] = []
            self.img2txt[i].extend(self.country_txt_ids[country])

            for txt_id in self.country_txt_ids[country]:
                self.txt2img[txt_id].append(i)
                
            self.image.append(im_name)

        self.text = self.text[:len(set(self.image)) * 2]

    def __getitem__(self, index):

        image_path = os.path.join(self.vis_root, self.annotation[index]["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        return {"image": image, "index": index}
    
class WGVRetrievalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """    
        self.vis_root = vis_root
        self.city_info = {}

        with open(ann_paths[0], 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                info = line[:-1].split(',')
                self.city_info[info[0]] = {'state' : info[1], 'country' : info[2], 'continent': info[3]}

        self.annotation = []
        txt_id = 0

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        image_files = os.listdir(self.vis_root)[:5000]

        for i, im_name in enumerate(image_files):
            self.annotation.append({'image' : im_name, 'instance_id' : str(i), 'caption' : []})
            city = im_name.split('_')[0]
            country = self.city_info[im_name.split('_')[0]]['country']
            texts = [f'a photo i took in {city}.', f'a photo showing the country of {country}.']

            self.img2txt[i] = []

            for text in texts:
                self.txt2img[txt_id] = i
                
                self.text.append(self.text_processor(text))
                self.annotation[i]['caption'].append(text)
                self.img2txt[i].append(txt_id)
                txt_id += 1

            self.image.append(im_name)


    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = self.text_processor(ann["caption"][0])

        return {
            "image": image,
            "text_input": caption,
            "image_id": index,
            "instance_id": ann["instance_id"],
        }