# WebKE: Knowledge Triple Extraction from Semi-structured Web with Pre-trained Markup Language Models

This repository contains code and data for the paper: **WebKE: Knowledge Triple Extraction from Semi-structured Web with Pre-trained Markup Language Models**.  Chenhao Xie, Wenhao Huang, Jiaqing Liang, Chengsong Huang and Yanghua Xiao. CIKM. 2021. [pdf](https://dl.acm.org/doi/10.1145/3459637.3482491)


## Files and Directory

- tokenization.py：tokenizer for html text.
- layers.py：overload `layers` module in bert4keras, mainly redefine the Embedding layers of Bert, for fusing layout imformation.
- models.py：overload `model` module in bert4keras, mainly redefine the input of Bert model so that it can take layout information.
- ljqpy.py：utils for loadling data and model.
- results/：weights for trained WebKE model.
- dataset/：preprocessed SWDE and expanded SWDE dataset.


## Model

- predicate_extraction_with_pos.py ：Open RE in phrase2.
- object_extraction_with_pos.py ：Open Object extraction in phrase2.
- segment_select_with_pos.py ：AOI Findings model in phrase1.

**Parameter description**

`load_model`：Boolean value indicating whether to load the pre-trained weights.

`maxlen`：maximum length of the input.

`bert_lock_layer`：the quantity of locked BERT layers, whose parameter won't change during training.

`lr`: learning rate.


## Main file

* html_extract_with_pos.py：Web extraction code using a two-step extraction model with layout information.

## Data Description

- *_with_pos.json: Preprocessed web data that every line contains a DOM tree node text with unlimited length and layout information.

## How to reproduce

1. Install all the dependencies in `requirements.txt`.
2. Download and unzip the [pretrained model](https://kw.fudan.edu.cn/resources/data/webke/result_tiny.zip) and put them in the `../pretrained_model/result_tiny` dolder.
3. Download and unzip the [prepocessed data](https://kw.fudan.edu.cn/resources/data/webke/webkedata.zip) and [weights](https://kw.fudan.edu.cn/resources/data/webke/webkeweights.zip) and put them into `dataset/` and `weight/`.
4. (Optional) Run `train.sh` to re-train the model, this will overwrite the files in `weight/`.
5. Run `html_extract_with_pos.py` to predict the test sites.


## Notice

1. Confirm that the `field_name` attribute in `predicate_extraction_with_pos.py`, `object_extraction_with_pos.py`, `segment_select_with_pos.py` and `html_extract_with_pos.py` remain same.
2. We only provided pretrained models with `tiny` size. In case other sizes is needed, you can contact the authors.

## Environments detail

NVIDIA-SMI 455.23.04

Driver Version: 455.23.04

CUDA Version: 11.1

GeForce RTX 3090

Python 3.7.9

Tensorflow 2.2.0

`requirements.txt` are provided for installing the virtual environment in conda.

## Citation
    @inproceedings{xie2021webke,
        title={WebKE: Knowledge Triple Extraction from Semi-structured Web with Pre-trained Markup Language Models.},
        author={Xie, Chenhao and Huang, Wenhao and Liang, Jiaqing and Huang, Chengsong and Xiao, Yanghua},
        booktitle={Proceedings of the 30th ACM International Conference on Information and Knowledge Management (CIKM)},
        year={2021}
    }
