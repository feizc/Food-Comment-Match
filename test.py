from PIL import Image 
import requests 
from model import FoodCommentModel, FoodFeatureExtractor
from transformers import BertTokenizer
import torch 

url = 'https://p0.meituan.net/ugcpic/c35b99aa4301d7a227dca0d2cd55a202'
image = Image.open(requests.get(url, stream=True).raw) 
img_extractor = FoodFeatureExtractor()
image = img_extractor(image)
model = FoodCommentModel.from_pretrained('ckpt/FoodComment') 
feature = model.get_image_features(image)
print(feature.size())

tokenizer = BertTokenizer.from_pretrained('ckpt/vocab.txt', do_lower_case=True)  
txt = '飞流直下三千尺，疑是银河落九天'
txt_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokenizer.tokenize(txt))]).long()
feature = model.get_text_features(txt_ids) 
print(feature.size())

