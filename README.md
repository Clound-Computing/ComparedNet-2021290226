# 2021290226


本文章主要代码来自"[Compare to The Knowledge: Graph Neural Fake News Detection with External Knowledge](https://aclanthology.org/2021.acl-long.62/)"
这一片文章

## 以下是该仓库的结构
```
FakeNewsDetection
├── README.md
├── *.py
└───models
|   └── *.py 
└───data
    ├── fakeNews
    │   ├── adjs
    │   │   ├── train
    │   │   ├── dev
    │   │   └── test
    │   ├── fulltrain.csv
    │   ├── balancedtest.csv
    │   ├── test.xlsx
    │   ├── entityDescCorpus.pkl
    │   ├── entity_feature_transE.pkl
    |   ├── gossipcop_v3-1_style_based_fake.csv
    |   └── gossipcop_v3-4_story_based_fake.csv
    └── stopwords_en.txt
└───LSTM(LSTM训练测试)
    ├── *.py
    └── gossipcop_v3-4_story_based_fake.csv
└───BiLSTM-BiCNN(BiLSTM以及CNN-BiLSTM训练测试)
    ├── *.py
    └── gossipcop_v3-4_story_based_fake.csv
└───BERT(BERT训练测试)
    ├── *.py
    ├── requirements.txt
    └── gossipcop_v3-4_story_based_fake.csv
└───preprocessing(处理课堂数据，使其转变为程序可用的格式)
    ├── *.py
    ├── ossipcop_v3-1_style_based_fake.json
    └── gossipcop_v3-4_story_based_fake.json
└───result
    └── *.xlsx
```


data中的更具体数据需要从以下链接下载: https://github.com/BUPT-GAMMA/CompareNet_FakeNewsDetection/releases/tag/dataset

## 环境
```
python 3.7
torch 1.8.1
nltk 3.2.5
tqdm
numpy
pandas
matplotlib
scikit_learn
xlrd 
```



## 运行

Train and test,
```
python main.py --mode 0
```

Test,
```
python main.py --mode 1 --model_file MODELNAME
```

Choose the different node to train and test(消融实验)
```
python main.py --mode 0 --node_type 0
```

```
node_type:
    '3 represents three types: Document&Entity&Topic;'
    '2 represents two types: Document&Entiy; '
    '1 represents two types: Document&Topic; '
    '0 represents only one type: Document. '
```



## 更改运行参数

```
python main.py  --mode 0
                --hidden_dim  # 隐藏层维度（默认100）
                --node_emb_dim  # 结点嵌入维度（默认32）
                --max_epochs  # 最大迭代次数（默认15）
                --ntags  # 数据集分类个数（在此次复现中选择的是2）
                --HALF  # 是否折半训练，节省内存
```

## 数据集

data文件夹中的gossipcop_v3-1_style_based_fake.csv以及gossipcop_v3-4_story_based_fake.csv为本次复现用到的数据集，可通过在启动时添加
```
python main.py  --mode 0 --train address
```
更改训练文件。

## 注意事项
由于课堂提供数据集有限，而该模型需要大量数据、知识库文件以及对应的图邻接矩阵才可实现较好的效果，本次复现过程中在使用新数据的同时使用了一部分的原数据，原数据链接在上面有提到