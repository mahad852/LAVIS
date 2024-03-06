import os
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess

device = torch.device("cuda")


root_image_dir = '/datasets/WGV/val'
labels_path = '/datasets/WGV/labels_list.csv'

image_files = os.listdir('/datasets/WGV/val')[:20]



# loads BLIP-2 pre-trained model
model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device)

city_info = {}

with open(labels_path, 'r') as f:
    lines = f.readlines()[1:]
    for line in lines:
        info = line[:-1].split(',')
        city_info[info[0]] = {'state' : info[1], 'country' : info[2], 'continent': info[3]}



open_ended_correct = 0
close_ended_correct = 0

popular_countries = ['united states', 'united kingdom', 'india']
default_choice_text = ','.join(popular_countries)

for i, im_name in enumerate(image_files):
    raw_image = Image.open(os.path.join(root_image_dir, im_name)).convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    pred_answer = model.generate({"image": image, 
                                  "prompt": "Question: which country is this image from? Answer:"}, use_nucleus_sampling=True, repetition_penalty=1.5)
    
    actual_answer = ' '.join(city_info['_'.join(im_name.split('_')[0:-2])]['country'].split('_')).lower()
    # print('model output1:', pred_answer)

    pred_answer = pred_answer[0].lower().split(',')[0]

    if pred_answer == actual_answer:
        open_ended_correct += 1

    print('image:', im_name, 'question: which country is this image from? answer:', pred_answer, 'actual answer:', actual_answer)
    
    
    choice_text = default_choice_text if actual_answer in popular_countries else default_choice_text + ',' + actual_answer
    
    pred_answer = model.generate({"image": image, 
                                  "prompt": f"Question: Which one of these countries ({choice_text}) is this image from? Answer:"},
                                  use_nucleus_sampling=True, repetition_penalty=1.5)
    
    # print('model output2:', pred_answer)

    pred_answer = pred_answer[0].lower().split(',')[0]
    if pred_answer == actual_answer or actual_answer in pred_answer:
        close_ended_correct += 1

    print('image:', im_name, f'Question: Which of these countries is this image from? ({choice_text}) answer:', pred_answer, 'actual answer:', actual_answer)
    print()
    
print('===================================================')
print('Open Ended Question accuracy:', open_ended_correct/len(image_files))
print('Close Ended Question accuracy:', close_ended_correct/len(image_files))
