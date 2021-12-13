from PIL import Image 
import requests 
from model import FoodCommentModel, FoodFeatureExtractor
from transformers import BertTokenizer
import torch 

import faiss 
import numpy as np 


import warnings
warnings.filterwarnings("ignore")

def image_encoding():
    url = 'https://p0.meituan.net/ugcpic/c35b99aa4301d7a227dca0d2cd55a202'
    image = Image.open(requests.get(url, stream=True).raw) 
    img_extractor = FoodFeatureExtractor()
    image = img_extractor(image)
    model = FoodCommentModel.from_pretrained('ckpt/FoodComment') 
    # print(model.state_dict()['vision_model.encoder.layers.1.layer_norm1.weight'])
    feature = model.get_image_features(image)
    print(feature.size())
    return feature.detach().numpy()

def text_encoding():
    tokenizer = BertTokenizer.from_pretrained('ckpt/vocab.txt', do_lower_case=True)  
    txt = '飞流直下三千尺，疑是银河落九天'
    model = FoodCommentModel.from_pretrained('ckpt/FoodComment') 
    txt_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokenizer.tokenize(txt))]).long()
    feature = model.get_text_features(txt_ids) 
    return feature.detach().numpy()


if __name__ == '__main__': 
    img_feature = image_encoding()
    txt_feature = text_encoding() 
    
    # construct search index 
    d = 512 
    index = faiss.IndexFlatL2(d) 
    index.add(img_feature) 

    k = 1
    D, I = index.search(txt_feature, k)



