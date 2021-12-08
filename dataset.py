import os 


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





if __name__ == '__main__': 
    data_path = 'data'
    img_txt_pair_list = get_data(data_path) 


