from object_extraction_with_pos import extract_objects
from predicate_extraction_with_pos import extract_preds
from segment_select_with_pos import extract_segments
from bs4 import BeautifulSoup,NavigableString,Tag
from bert4keras.tokenizers import Tokenizer
from tokenization import Tokenizer1
import os,sys,json
from tqdm import tqdm
import random

tot_node = 0
sel_node = 0 
node_list = []
interest_node_list = []
maxlen = 512
sample_ratio = 1
use_segment_selector = False
field_name = 'nbaplayer'

tokenizer = Tokenizer1('vocab.txt', do_lower_case=True)

def load_data(filename):
    """加载数据
    """
    D = []
    H = []
    tag = [False] * 500
    if isinstance(filename,list):
        for filename1 in filename:
            with open(filename1, encoding='utf-8') as f:
                for l in tqdm(f, desc='load data'):
                    l = json.loads(l)
                    if tag[l['index']]:
                        D = sorted(D, key=lambda x:x['index'])
                        H.append(D)
                        tag = [False] * 500
                        D = []
                    tag[l['index']] = True
                    D.append(l)
            H.append(D)
            D = []
    else:
        with open(filename, encoding='utf-8') as f:
            for l in tqdm(f, desc='load data'):
                l = json.loads(l)
                
                if tag[l['index']]:
                    D = sorted(D, key=lambda x:x['index'])
                    H.append(D)
                    tag = [False] * 500
                    D = []
                tag[l['index']] = True
                D.append(l)
        H.append(D)
        D = []
    return H

data = load_data([f'dataset/test_{field_name}_with_pos.json'])

def data_split(full_list, ratio, shuffle=False):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     子列表1
    :param shuffle:   子列表2
    :return:
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2

def get_filelist(path):
    filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
        # 文件名列表，包含完整路径
            filelist.append(os.path.join(home, filename))
    return filelist

def attrs_to_str(res):
    ret = ''
    for key,value in res.items():
        ret

def pair_judge(p1, p2):
    if (p1[0] == p2[0] + 1) and (p1[1] == p2[1] + 1):
        return 0
    elif (p1[1] < p2[0] + 1):
        return -1
    elif (p1[0] > p2[1] + 1):
        return 1

def html_extract(data):
    global interest_node_list
    global node_list
    res = []
    interest = [False] * 500
    child = [False] * 500
    print(len(data))
    for i,h in enumerate(data):
        if (not use_segment_selector) or (not child[i]) or interest[i]:
            for x in h['children']:
                child[x[2]] = True
            #print(string)
            children_list = h['children']
            if len(children_list) > 0:
                segments_list = extract_segments(h['text'],h['pos'])
                p = 0
                for item in segments_list:
                    while (p < len(children_list)) and (pair_judge(item,children_list[p][1]) == 1):
                        #print(children_list[p][1])
                        p = p + 1 
                    if (p < len(children_list)) and pair_judge(item,children_list[p][1]) == 0:
                        #segment_controller(children_list[pos][1])
                        interest[children_list[p][2]] = True
            string = h['text']
            pos = h['pos']
            preds_list = extract_preds(string, pos)
            # print(preds_list)
            #res.extend(preds_list)
            
            for pred in preds_list:
                object_list = extract_objects(pred, string, pos)
                for objectx in object_list:
                    res.append((pred,objectx))
    return res

    
def evaluate(data):
    """评估函数，计算f1、precision、recall
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open(f'results/dev_{field_name}.json', 'w', encoding='utf-8')
    pbar = tqdm()
    num = int(len(data) * sample_ratio)
    print(num)
    data = random.sample(data, num)
    for d in data:
        R = set([pred for pred in html_extract(d)])
        #T = set([pred for pred in d['spo_list']])
        T = []
        for dd in d:
            for k, v in dd['spo_list'].items():
                for vv in v:
                    T.append((k,vv))
                    #T.append(k)
        T = set(T)
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()
        pbar.set_description(
            'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
        )
        s = json.dumps({
            #'text': d['text'],
            'pred_list': list(T),
            'pred_list_pred': list(R),
            'new': list(R - T),
            'lack': list(T - R),
        },
                       ensure_ascii=False,
                       indent=4)
        f.write(s + '\n')
    pbar.close()
    f.close()
    return f1, precision, recall




if __name__ == '__main__':
    evaluate(data)
    #print(sel_node)
    #print(tot_node)
