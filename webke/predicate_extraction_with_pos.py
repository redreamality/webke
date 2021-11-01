import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
os.environ['TF_KERAS'] = '1'
import json
import numpy as np
from tokenization import Tokenizer1
from bert4keras.backend import keras, K, batch_gather
from layers import Loss
from layers import LayerNormalization,Embedding
from bert4keras.tokenizers import Tokenizer
# from bert4keras.models import build_transformer_model
from models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_exponential_moving_average
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, to_array
from keras.layers import Input, Dense, Lambda, Reshape
from keras.models import Model
from tqdm import tqdm
import random
import re

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

start_t = 0.6
end_t = 0.5
load_model = False
maxlen = 512
batch_size = 4
config_path = config_dict[bert_model]
checkpoint_path = checkpoint_dict[bert_model]
dict_path = 'vocab.txt'
bert_lock_layer = 0
field_name = 'nbaplayer'
lr = 2e-5
test_ratio = 0.1

def load_data(filename):
    """加载数据
    """
    D = []
    if isinstance(filename,list):
        for filename1 in filename:
            with open(filename1, encoding='utf-8') as f:
                for l in tqdm(f, desc='load data'):
                    l = json.loads(l)
                    if len(l['spo_list'].keys())>0:
                        D.append({
                            'text': l['text'].replace('\ufeff',''),
                            'pred_list': l['spo_list'].keys(),
                            'pos': l['pos'],
                        })
    else:
        with open(filename, encoding='utf-8') as f:
            for l in tqdm(f, desc='load data'):
                l = json.loads(l)
                if len(l['spo_list'].keys())>0:
                    D.append({
                        'text': l['text'].replace('\ufeff',''),
                        'pred_list': l['spo_list'].keys(),
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
savemodel_path = f'weight/{bert_model}_predicate_{field_name}_with_pos.weights' 

#print(len(tokenizer.tokenize(train_data[4]['text'])))
#print(len(train_data[4]['pos']))

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
        batch_pred_labels = []
        for is_end, d in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(d['text'])
            #print(len(token_ids))
            #print(len(d['pos']) + 2)
            preds = []
            for pred in d['pred_list']:
                pred = tokenizer.encode(pred)[0][1:-1]
                pred_idx = search(pred, token_ids)
                if pred_idx != -1:
                    pred = (pred_idx, pred_idx + len(pred) - 1)
                    preds.append(pred)
            if preds:
                # pred标签
                pred_labels = np.zeros((len(token_ids),2))
                for pred in preds:
                    #print(pred)
                    pred_labels[pred[0], 0] = 1
                    pred_labels[pred[1], 1] = 1
                # 构建batch
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_pred_labels.append(pred_labels)
                batch_x0.append([0] + [i[0] for i in d['pos']] + [0])
                batch_y0.append([0] + [i[1] for i in d['pos']] + [0])
                batch_x1.append([0] + [i[2] for i in d['pos']] + [0])
                batch_y1.append([0] + [i[3] for i in d['pos']] + [0])
                assert len(token_ids) == len(d['pos']) + 2
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_pred_labels = sequence_padding(batch_pred_labels)
                    batch_x0 = sequence_padding(batch_x0)
                    batch_y0 = sequence_padding(batch_y0)
                    batch_x1 = sequence_padding(batch_x1)
                    batch_y1 = sequence_padding(batch_y1)
                    
                    #batch_pred_ids = np.array(batch_subject_ids)
                    #batch_object_labels = sequence_padding(batch_object_labels)
                    yield [
                        batch_token_ids, batch_segment_ids, batch_x0, batch_y0, batch_x1, batch_y1
                    ], batch_pred_labels
                    batch_token_ids, batch_segment_ids = [], []
                    batch_x0, batch_y0, batch_x1, batch_y1 = [], [], [], []
                    batch_pred_labels = []

#data = data_generator(train_data, batch_size)
#for d in data:
#    print(d)


# 补充输入
#pred_labels = Input(shape=(,4), name='Subject-Labels')

# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
    hierarchical_position=True,
    custom_position_ids=True
)

output = Dense(
    2, activation='sigmoid', kernel_initializer=bert.initializer
)(bert.model.output)
#pred_preds = Lambda(lambda x: x**2)(output)

pred_model = Model(bert.model.inputs, output)
#AdamEMA = extend_with_exponential_moving_average(Adam, name='AdamEMA')
#optimizer = AdamEMA(learning_rate=1e-5)
pred_model.compile(loss='binary_crossentropy', 
                    optimizer=Adam(lr),
                    metrics=['accuracy'],)

lock_namelist = ['Transformer-'+str(i) for i in range(bert_lock_layer)]
for layer in pred_model.layers:
    layerName=str(layer.name)
    for name in lock_namelist:
        if layerName.startswith(name):
            layer.trainable = False
            break

#pred_model.summary()



def extract_preds(text, pos):
    """抽取输入text所包含的三元组
    """
    tokens = tokenizer.tokenize(text)
    mapping = tokenizer.rematch(text, tokens)
    token_ids, segment_ids = tokenizer.encode(text)
    token_ids, segment_ids = to_array([token_ids], [segment_ids])
    x0,y0,x1,y1 = to_array([[0] + [i[0] for i in pos] + [0]], \
                            [[0] + [i[1] for i in pos] + [0]], \
                            [[0] + [i[2] for i in pos] + [0]], \
                            [[0] + [i[3] for i in pos] + [0]])
    # 抽取subject
    pred_preds = pred_model.predict([token_ids, segment_ids, x0, y0, x1, y1])
    start = np.where(pred_preds[0, :, 0] > start_t)[0]
    end = np.where(pred_preds[0, :, 1] > end_t)[0]
    
    #print(pred_preds)
    preds = []
    for i in start:
        j = end[end >= i]
        if len(j) > 0:
            j = j[0]
            preds.append((i, j))
    #print([text[mapping[pred[0]][0]:mapping[pred[1]][-1]+1] for pred in preds if len(mapping[pred[0]]) != 0 and len(mapping[pred[1]]) != 0])
    ret = [text[mapping[pred[0]][0]:mapping[pred[1]][-1] + 1] for pred in preds if len(mapping[pred[0]]) != 0 and len(mapping[pred[1]]) != 0]
    ret = [string for string in ret if re.findall('<.*>',string) == []]
    return ret





def evaluate(data):
    """评估函数，计算f1、precision、recall
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open('results/dev_pred_with_pos.json', 'w', encoding='utf-8')
    pbar = tqdm()
    for d in data:
        R = set([pred for pred in extract_preds(d['text'],d['pos'])])
        T = set([pred for pred in d['pred_list']])
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


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = evaluate(valid_data)
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            pred_model.save_weights(savemodel_path)
        print(
            'f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )
        self.file_stream.write('%s %s %d :\tf1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (field_name, bert_model, epoch, f1, precision, recall, self.best_val_f1))
    
    def on_train_begin(self, logs=None):
        self.file_stream = open('weight/record.txt','a')
        self.file_stream.write('%s %s %s %f\n'%(field_name, bert_model, 'predicate', lr))
        self.file_stream.write('*'*70 + '\n')

    def on_train_end(self, logs=None):
        self.file_stream.write('\n')
        self.file_stream.close()


if __name__ == '__main__':
    # 加载数据集
    data = load_data([f'dataset/train_{field_name}_with_pos.json'])
    train_data, valid_data = data_split(data, ratio=0.9, shuffle=True)

    train_generator = data_generator(train_data, batch_size)
    evaluator = Evaluator()
    if os.path.exists(savemodel_path + '.index') and load_model:
        pred_model.load_weights(savemodel_path)
    pred_model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=10,
        callbacks=[evaluator]
    )

else:

    pred_model.load_weights(savemodel_path)
