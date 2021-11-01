import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
os.environ['TF_KERAS'] = '1'
import json
import numpy as np
from tokenization import Tokenizer1
from bert4keras.backend import keras, K, batch_gather
from layers import Loss
from layers import LayerNormalization
from bert4keras.tokenizers import Tokenizer
from models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_exponential_moving_average
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, to_array
from keras.layers import Input, Dense, Lambda, Reshape, Softmax
from keras.models import Model
from tqdm import tqdm
import random
import re

start_t = 0.6
end_t = 0.5
bert_model = 'tiny'
config_dict = {
    'tiny': '../pretrained_model/result_tiny/bert_config.json',
    'base': '../pretrained_model/result/bert_config.json',
    'mini': '../pretrained_model/result_mini/bert_config.json',
    'small': '../pretrained_model/result_small/bert_config.json',
}
checkpoint_dict = {
    'tiny': '../pretrained_model/result_tiny',
    'base': '../pretrained_model/result',
    'mini': '../pretrained_model/result_mini',
    'small': '../pretrained_model/result_small',
}

load_model = False
maxlen = 512
batch_size = 4
config_path = config_dict[bert_model]
checkpoint_path = checkpoint_dict[bert_model]
dict_path = 'vocab.txt'
bert_lock_layer = 6
field_name = 'nbaplayer'
lr = 2e-5
test_ratio = 0.1
#max_len:  1067
#sum_len:  214


def load_data(filename):
    """加载数据
    单条格式：{'text': text, 'spo_list': [(s, p, o)]}
    """
    D = []
    if isinstance(filename,list):
        for filename1 in filename:
            with open(filename1, encoding='utf-8') as f:
            
                num = len
                for index, l in tqdm(enumerate(f), desc='load data'):
                    l = json.loads(l)
                    for key, value in l['spo_list'].items():
                        D.append({
                            'query': key,
                            'text': l['text'],
                            'object': value,
                            'pos': l['pos'],
                        })
    else:
        with open(filename, encoding='utf-8') as f:
            for l in tqdm(f, desc='load data'):
                l = json.loads(l)
                for key, value in l['spo_list'].items():
                    D.append({
                        'query': key,
                        'text': l['text'],
                        'object': value,
                        'pos': l['pos'],
                    })
    return D

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


# 建立分词器
tokenizer = Tokenizer1(dict_path, do_lower_case=True)
savemodel_path = f'weight/{bert_model}_object_{field_name}_with_pos.weights' 

def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        batch_x0, batch_y0, batch_x1, batch_y1 = [], [], [], []
        batch_object_labels = []
        for is_end, d in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(d['query'],d['text'])
            objects = []
            for objectx in d['object']:
                objectx = tokenizer.encode(objectx)[0][1:-1]
                object_idx = search(objectx, token_ids)
                if object_idx != -1:
                    objectx = (object_idx, object_idx + len(objectx) - 1)
                    objects.append(objectx)

            if objects:
                object_labels = np.zeros((len(token_ids), 2))
                for objectx in objects:
                    object_labels[objectx[0], 0] = 1
                    object_labels[objectx[1], 1] = 1
                # 构建batch
                query_token_len = len(tokenizer.tokenize(d['query'])) - 2
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_object_labels.append(object_labels)
                batch_x0.append([0] + [0] * (query_token_len + 1) + [i[0] for i in d['pos']] + [0])
                batch_y0.append([0] + [0] * (query_token_len + 1) + [i[1] for i in d['pos']] + [0])
                batch_x1.append([0] + [0] * (query_token_len + 1) + [i[2] for i in d['pos']] + [0])
                batch_y1.append([0] + [0] * (query_token_len + 1) + [i[3] for i in d['pos']] + [0])
                assert len(token_ids) == len(d['pos']) + 3 + query_token_len
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_object_labels = sequence_padding(batch_object_labels)
                    batch_x0 = sequence_padding(batch_x0)
                    batch_y0 = sequence_padding(batch_y0)
                    batch_x1 = sequence_padding(batch_x1)
                    batch_y1 = sequence_padding(batch_y1)
                    #print(batch_object_labels)
                    yield [
                        batch_token_ids, batch_segment_ids, batch_x0, batch_y0, batch_x1, batch_y1
                    ], batch_object_labels
                    batch_x0, batch_y0, batch_x1, batch_y1 = [], [], [], []
                    batch_token_ids, batch_segment_ids = [], []
                    batch_object_labels = []



# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
    hierarchical_position=True,
    custom_position_ids=True
)

# 预测object

output = Dense(
    2, activation='sigmoid', kernel_initializer=bert.initializer
)(bert.model.output) # [batch,seq_len,2]

object_model = Model(bert.model.inputs, output)
object_model.compile(loss='binary_crossentropy', 
                    optimizer=Adam(lr),
                    metrics=['accuracy'],)

lock_namelist = ['Transformer-'+str(i) for i in range(bert_lock_layer)]
for layer in object_model.layers:
    layerName=str(layer.name)
    for name in lock_namelist:
        if layerName.startswith(name):
            layer.trainable = False
            break

#object_model.summary()


def extract_objects(query, text, pos):
    """抽取输入text和query所对应的属性值
    """
    
    origin_text = query + ' ' + text
    token_ids, segment_ids = tokenizer.encode(query, text)
    tokens = [tokenizer.id_to_token(token_id) for token_id in token_ids]
    query_token_len = len(tokenizer.tokenize(query)) - 2
    mapping = tokenizer.rematch(origin_text, tokens)
    token_ids, segment_ids = to_array([token_ids], [segment_ids])
    x0,y0,x1,y1 = to_array([[0] + [0] * (query_token_len + 1) + [i[0] for i in pos] + [0]], \
                            [[0] + [0] * (query_token_len + 1) + [i[1] for i in pos] + [0]], \
                            [[0] + [0] * (query_token_len + 1) + [i[2] for i in pos] + [0]], \
                            [[0] + [0] * (query_token_len + 1) + [i[3] for i in pos] + [0]])
    # 抽取object
    query_ids = tokenizer.tokenize(query)[1:-1]
    id = search(query_ids, token_ids)
    object_preds = object_model.predict([token_ids, segment_ids, x0, y0, x1, y1])
    start = np.where(object_preds[0, :, 0] > start_t)[0]
    end = np.where(object_preds[0, :, 1] > end_t)[0]
    objects = []
    for i in start:
        for j in end:
        #j = end[end >= i and end >= id]
            if j > i and j > id:
            #if len(j) > 0:
                #j = j[0]
                objects.append((i, j))
                break
    ret = [origin_text[mapping[objectx[0]][0]:mapping[objectx[1]][-1] + 1] for objectx in objects if len(mapping[objectx[0]]) != 0 and len(mapping[objectx[1]]) != 0]
    ret = [string for string in ret if re.findall('<.*>',string) == []]
    return ret



def evaluate(data):
    """评估函数，计算f1、precision、recall
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open('results/dev_obj_with_pos.json', 'w', encoding='utf-8')
    pbar = tqdm()
    for d in data:
        R = set([objectx for objectx in extract_objects(d['query'],d['text'],d['pos'])])
        T = set(d['object'])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()
        pbar.set_description(
            'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
        )
        s = json.dumps({
            'text': d['text'],
            'predicate': d['query'],
            'object_list': list(T),
            'object_list_pred': list(R),
            'new': list(R - T),
            'lack': list(T - R),
        },
                       ensure_ascii=False,
                       indent=4)
        f.write(s + '\n')
    pbar.close()
    f.close()
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = evaluate(valid_data)
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            object_model.save_weights(savemodel_path)
        print(
            'f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )
        self.file_stream.write('%s %s %d :\tf1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (field_name, bert_model, epoch, f1, precision, recall, self.best_val_f1))
    
    def on_train_begin(self, logs=None):
        self.file_stream = open('weight/record.txt','a')
        self.file_stream.write('%s %s %s %f\n'%(field_name, bert_model, 'object', lr))
        self.file_stream.write('*'*70 + '\n')

    def on_train_end(self, logs=None):
        self.file_stream.write('\n')
        self.file_stream.close()

if __name__ == '__main__':
    # 加载数据集
    data = load_data([f'dataset/train_{field_name}_with_pos.json'])

    train_data, valid_data = data_split(data, ratio=0.95, shuffle=True)

    train_generator = data_generator(train_data, batch_size)
    evaluator = Evaluator()
    if os.path.exists(savemodel_path+'.index') and load_model:
        object_model.load_weights(savemodel_path)
    #evaluate(valid_data)
    object_model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=20,
        callbacks=[evaluator]
    )
    
else:
    print(savemodel_path)
    object_model.load_weights(savemodel_path)
