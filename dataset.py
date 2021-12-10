import os 
from transformers import BertTokenizer 
from torch.utils.data import Dataset 
from PIL import Image 
import requests 
from model import FoodFeatureExtractor 
import torch 


def tokenize(obj,tokenizer):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o)) for n, o in obj.items())
    return list(tokenize(o) for o in obj) 

def get_data(data_path):
    file_name_list = os.listdir(data_path) 
    img_txt_pair_list = []
    for file_name in file_name_list: 
        file_path = os.path.join(data_path, file_name)
        if 'part-' not in file_path:
            continue
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines[:100]: 
            elements = line.strip().split('\t')
            img_txt = {}  
            if len(elements) == 3:
                img_txt['url'] = url_supplement(elements[1])
                img_txt['txt'] = elements[0]
            else:
                img_txt['url'] = url_supplement(elements[2])
                img_txt['txt'] = elements[1] 
            img_txt_pair_list.append(img_txt)
    return img_txt_pair_list 


def url_supplement(raw_url): 
    if 'http' not in raw_url:
        url = 'https://p0.meituan.net/ugcpic/' + raw_url 
    else:
        url = raw_url 
    return url 


class FoodCommentDataset(Dataset): 
    def __init__(self, img_txt_pair_list, tokenizer, img_extractor):
        self.img_txt_pair_list = img_txt_pair_list 
        self.tokenizer = tokenizer 
        self.img_extractor = img_extractor 
    
    def __len__(self):
        return len(self.img_txt_pair_list) 
    
    def __getitem__(self, index):
        img_txt_pair = self.img_txt_pair_list[index] 
        url = img_txt_pair['url'] 
        txt = img_txt_pair['txt'] 
        image = Image.open(requests.get(url, stream=True).raw) 
        image = self.img_extractor(image).squeeze(0)
        txt_ids = torch.Tensor(tokenize(txt, self.tokenizer)).long() 
        return image, txt_ids  

if __name__ == '__main__': 
    data_path = 'data'
    img_txt_pair_list = get_data(data_path) 
    tokenizer = BertTokenizer('ckpt/vocab.txt', do_lower_case=True) 
    img_extractor = FoodFeatureExtractor()
    dataset = FoodCommentDataset(img_txt_pair_list, tokenizer, img_extractor) 
    print(dataset[0])

