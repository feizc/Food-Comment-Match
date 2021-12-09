import os 
from transformers import BertTokenizer 
from torch.utils.data import Dataset 


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
        url = 'https://p0.meituan.net/coverpic/' + raw_url 
    else:
        url = raw_url 
    return url 


class FoodCommentDataset(Dataset): 
    def __init__(self, img_txt_pair_list, tokenizer):
        self.img_txt_pair_list = img_txt_pair_list 
        self.tokenizer = tokenizer 
    
    def __len__(self):
        return len(self.img_txt_pair_list) 
    
    def __getitem__(self, index):
        img_txt_pair = self.img_txt_pair_list[index] 
        url = img_txt_pair['url'] 
        txt = img_txt_pair['txt'] 
        return url, txt 
        



if __name__ == '__main__': 
    data_path = 'data'
    img_txt_pair_list = get_data(data_path) 
    tokenizer = BertTokenizer('ckpt/vocab.txt', do_lower_case=True) 
    dataset = FoodCommentDataset(img_txt_pair_list, tokenizer) 
    print(dataset[1])


