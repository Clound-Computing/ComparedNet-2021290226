本文章主要代码来自"[Compare to The Knowledge: Graph Neural Fake News Detection with External Knowledge](https://aclanthology.org/2021.acl-long.62/)"
这一片文章

以下是该仓库的结构
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

# 环境
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



# 运行

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
python main.py --mode 0 --node 0
```

```
node_type:
    '3 represents three types: Document&Entity&Topic;'
    '2 represents two types: Document&Entiy; '
    '1 represents two types: Document&Topic; '
    '0 represents only one type: Document. '
```