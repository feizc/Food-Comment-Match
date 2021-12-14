from torch.utils.data.dataloader import DataLoader
from transformers import BertTokenizer, AdamW
import torch 
import numpy as np 
import random 
from torch.utils.data import DataLoader
from model import FoodFeatureExtractor, FoodCommentConfig, FoodCommentModel, contrastive_loss
from argparse import ArgumentParser 
from dataset import get_data, FoodCommentDataset, collate_fn
import os 
from configuration import FoodCommentConfig 

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
    running_acc = .0
    with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(data_loader)) as pbar: 
        for it, (img, txt) in enumerate(data_loader): 
            img, txt = img.to(device), txt.to(device) 
            output = model(input_ids=txt, pixel_values=img, return_loss=True) 
            loss = output[0]
            acc = output[1].item()
            loss.backward()

            optimizer.step() 
            this_loss = loss.item() 
            running_loss += this_loss 
            running_acc += acc 

            pbar.set_postfix(loss=running_loss / (it + 1), acc=running_acc / (it + 1)) 
            pbar.update() 
    loss = running_loss / len(data_loader) 
    return loss 


if __name__ == '__main__': 
    parser = ArgumentParser() 
    parser.add_argument('--data_path', type=str, default='data') 
    parser.add_argument('--model_path', type=str, default='ckpt/original') 
    parser.add_argument('--save_path', type=str, default='ckpt/FoodComment')
    parser.add_argument('--use_ckpt', type=bool, default=False)
    args = parser.parse_args() 

    print(args) 
    use_cuda = torch.cuda.is_available() 
    device = torch.device('cuda' if use_cuda else 'cpu') 
    lr = 6e-3 
    batch_size = 8
    epochs = 8 

    tokenizer_class = BertTokenizer 
    model_class = FoodCommentModel 
    if args.use_ckpt == True: 
        model = model_class.from_pretrained(args.save_path) 
        tokenizer = tokenizer_class.from_pretrained(args.save_path)
        ckpt = torch.load('ckpt/FoodComment/pytorch_model.bin', map_location='cpu')
        model.load_state_dict(ckpt) 

    else:
        configure = FoodCommentConfig()
        tokenizer = tokenizer_class.from_pretrained('ckpt/vocab.txt', do_lower_case=True) 
        model = model_class(configure) 
    pad_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('[PAD]'))[0]

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr) 
    # optimizer.load_state_dict(ckpt['optimizer'])

    train_data = get_data(args.data_path)
    img_extractor = FoodFeatureExtractor() 
    train_dataset = FoodCommentDataset(train_data, tokenizer, img_extractor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=lambda x: collate_fn(x, pad_token))  
    

    for epoch in range(epochs): 
        train_loss = train(train_loader, model, optimizer, epoch) 

        # save checkpoint
        torch.save(model.state_dict(), '%s/pytorch_model.bin'%(args.save_path))
        model.config.to_json_file(os.path.join(args.save_path, 'config.json'))
        tokenizer.save_vocabulary(args.save_path) 
        





