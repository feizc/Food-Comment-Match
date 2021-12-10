from torch.utils.data.dataloader import DataLoader
from transformers import BertTokenizer, AdamW
import torch 
import numpy as np 
import random 
from torch.utils.data import DataLoader
from model import FoodFeatureExtractor, FoodCommentConfig, FoodCommentModel, contrastive_loss
from argparse import ArgumentParser 
from dataset import get_data, FoodCommentDataset 

import warnings
warnings.filterwarnings("ignore")


from PIL import Image 
import requests 
from tqdm import tqdm 



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(2021)


def train(data_loader, model, optimizer, epoch):
    model.train() 
    running_loss = .0 
    with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(data_loader)) as pbar: 
        for it, (img, txt) in enumerate(data_loader): 
            img, txt = img.to(device), txt.to(device) 
            loss = model(input_ids=txt, pixel_values=img, return_loss=True)[0] 
            loss.backward()

            optimizer.step() 
            this_loss = loss.item() 
            running_loss += this_loss 

            pbar.set_postfix(loss=running_loss / (it + 1)) 
            pbar.update() 
    loss = running_loss / len(data_loader) 
    return loss 





if __name__ == '__main__': 
    parser = ArgumentParser() 
    parser.add_argument('--data_path', type=str, default='data') 
    parser.add_argument('--model_path', type=str, default='ckpt/original') 
    args = parser.parse_args() 

    print(args) 
    use_cuda = torch.cuda.is_available() 
    device = torch.device('cuda' if use_cuda else 'cpu') 
    lr = 6e-5 
    batch_size = 1
    epochs = 1 

    tokenizer_class = BertTokenizer 
    tokenizer = tokenizer_class.from_pretrained('ckpt/vocab.txt', do_lower_case=True) 
    img_extractor = FoodFeatureExtractor() 
    model_class = FoodCommentModel
    model = model_class.from_pretrained('ckpt/original') 

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr) 

    train_data = get_data(args.data_path)
    train_dataset = FoodCommentDataset(train_data, tokenizer, img_extractor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)  
    

    for epoch in range(epochs): 
        train(train_loader, model, optimizer, epoch)


