#!/usr/bin/env python
# coding: utf-8

# 使用するライブラリのインストールや、データの前処理などを管理するファイルです。

# In[11]:


get_ipython().system('jupyter nbconvert --to script data_admin.ipynb')


# In[1]:


# ライブラリのインポート


# In[2]:


import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import os , shutil, pathlib
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint


# In[3]:


# cat_dogファイルの画像をtrain, validation, testの3つのディレクトリにコピー


# In[4]:


# 元データのパズ
original_dir=pathlib.Path('/Users/iinoshunhei/Desktop/cat-and-dog')
#3つのサブセットを格納するディレクトリへのパス
new_base_dir=pathlib.Path('/Users/iinoshunhei/Desktop/small_data')


# In[5]:


# 各データファイルの作成関数
def make_subset(subset_name, start_index, end_index): 
    for category in ("cat", "dog"):
        dir= new_base_dir/subset_name/category
        os.makedirs(dir)
        fnames= [f"{category}.{i}.jpg" for i in range(start_index, end_index)]
        print(fnames)
        for fname in fnames:
            shutil.copyfile(src=original_dir/fname, dst=dir/fname)


# In[6]:


# ファイルの有無で実行するか判断する関数
def file_exist_check(subset_name, start_index, end_index):
    directory='/Users/iinoshunhei/Desktop/small_data'
    filename=subset_name
    file_path=directory + '/' + filename

    if not os.path.exists(file_path):
        make_subset('train', start_index=start_index, end_index=end_index)
    else:
        pass


# In[7]:


#　各データのファイル作成
file_exist_check('train', 1, 1000)
file_exist_check('validation', 1000, 1500)
file_exist_check('test', 1500, 2500)


# In[8]:


# データの前処理を行う(image_dataset_from_dictionary())
from tensorflow.keras.utils import image_dataset_from_directory
# バッチの数＝テータ数/バッチ数
train_dataset=image_dataset_from_directory(new_base_dir/'train', image_size=(180,180), batch_size=32)
validation_dataset=image_dataset_from_directory(new_base_dir/'validation', image_size=(180,180), batch_size=32)
test_dataset=image_dataset_from_directory(new_base_dir/'test', image_size=(180,180), batch_size=32)


# In[9]:


# データを正則化するために、範囲を確認
for batch_data, _ in train_dataset:
    min=np.min(batch_data)
    max=np.max(batch_data)
    print(min)
    print(max)   


# In[10]:


# datasetが生成するデータの形状とラベルを表示
for data_batch, labels_batch in train_dataset:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
#     print(labels_batch)を実行すると、ランダムなバッチひとつの状態が返される
    break


# In[ ]:




