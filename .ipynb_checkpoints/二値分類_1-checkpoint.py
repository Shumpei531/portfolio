#!/usr/bin/env python
# coding: utf-8

# 犬と猫の画像の二値分類をCNNにより作成しました。
# 具体的には、少量の訓練データ、評価データ、テストデータを作成し、訓練と推論を行いました。
# また精度を向上させるため、データの拡張及びドロップアウトを追加して同様に行いました。

# In[ ]:


get_ipython().system('jupyter nbconvert --to script 二値分類_1.ipynb')


# In[ ]:


# ライブラリのインポート


# In[20]:


import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


# In[4]:


# cat_dogファイルの画像をtrain, validation, testの3つのディレクトリにコピー


# In[5]:


import os , shutil, pathlib


# In[6]:


# 元データのパズ
original_dir=pathlib.Path('/Users/iinoshunhei/Desktop/cat-and-dog')
#3つのサブセットを格納するディレクトリへのパス
new_base_dir=pathlib.Path('/Users/iinoshunhei/Desktop/small_data')


# In[7]:


# 各データファイルの作成関数
def make_subset(subset_name, start_index, end_index): 
    for category in ("cat", "dog"):
        dir= new_base_dir/subset_name/category
        os.makedirs(dir)
        fnames= [f"{category}.{i}.jpg" for i in range(start_index, end_index)]
        print(fnames)
        for fname in fnames:
            shutil.copyfile(src=original_dir/fname, dst=dir/fname)


# In[24]:


# ファイルの有無で実行するか判断する関数
def file_exist_check(subset_name, start_index, end_index):
    directory='/Users/iinoshunhei/Desktop/small_data'
    filename=subset_name
    file_path=directory + '/' + filename

    if not os.path.exists(file_path):
        make_subset('train', start_index=start_index, end_index=end_index)
    else:
        pass


# In[27]:


#　各データのファイル作成
file_exist_check('train', 1, 1000)
file_exist_check('validation', 1000, 1500)
file_exist_check('test', 1500, 2500)


# In[11]:


# 犬と猫を分類するためのcnnをインスタンス化
inputs=keras.Input(shape=(180,180,3))
x=layers.Rescaling(1./255)(inputs)
x=layers.Conv2D(filters=32, kernel_size=3, activation='relu')(inputs)
x=layers.MaxPooling2D(pool_size=2)(x)
x=layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
x=layers.MaxPooling2D(pool_size=2)(x)
x=layers.Conv2D(filters=128, kernel_size=3, activation='relu')(x)
x=layers.MaxPooling2D(pool_size=2)(x)
x=layers.Conv2D(filters=256, kernel_size=3, activation='relu')(x)
x=layers.MaxPooling2D(pool_size=2)(x)
x=layers.Conv2D(filters=256, kernel_size=3, activation='relu')(x)
x=layers.Flatten()(x)
outputs=layers.Dense(1, activation='sigmoid')(x)
model=keras.Model(inputs=inputs, outputs=outputs)


# In[12]:


model.summary()


# In[13]:


# モデルのコンパイル
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])


# In[18]:


# データの前処理を行う(image_dataset_from_dictionary())
from tensorflow.keras.utils import image_dataset_from_directory
# バッチの数＝テータ数/バッチ数
train_dataset=image_dataset_from_directory(new_base_dir/'train', image_size=(180,180), batch_size=32)
validation_dataset=image_dataset_from_directory(new_base_dir/'validation', image_size=(180,180), batch_size=32)
test_dataset=image_dataset_from_directory(new_base_dir/'test', image_size=(180,180), batch_size=32)


# In[15]:


# データを正則化するために、範囲を確認
for batch_data, _ in train_dataset:
    min=np.min(batch_data)
    max=np.max(batch_data)
    print(min)
    print(max)   


# In[16]:


# datasetが生成するデータの形状とラベルを表示
for data_batch, labels_batch in train_dataset:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
#     print(labels_batch)を実行すると、ランダムなバッチひとつの状態が返される
    break


# In[42]:


# datasetを用いてモデルの適合
from keras.callbacks import ModelCheckpoint
callbacks=[
    ModelCheckpoint(filepath="convnet_cat&dog.h5",
                                   save_best_only=True,
                                   monitor='val_loss',period=1)
]
history=model.fit(train_dataset, epochs=30, validation_data=validation_dataset, callbacks=[callbacks])


# In[44]:


# 訓練時の損失値と正解率をプロット
import matplotlib.pyplot as plt
accuracy=history.history["accuracy"]
val_accuracy=history.history["val_accuracy"]
loss=history.history["loss"]
val_loss=history.history["val_loss"]
epochs=range(1, len(accuracy)+1)
plt.figure()
plt.plot(epochs, accuracy, 'bo', label="training_accuracy")
plt.plot(epochs, val_accuracy, 'b', label="validation_accuracy")
plt.title("training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label="training_loss")
plt.plot(epochs, val_loss, 'b', label="validation_loss")
plt.title("training and validation loss")
plt.legend()
plt.show()


# In[43]:


# テストデータでモデルの評価(1回目)


# In[46]:


test_model=keras.models.load_model("convnet_cat&dog.h5")
test_loss, test_acc=test_model.evaluate(test_dataset)
print(f"test accuracy : {test_acc:.3f}")


# In[47]:


# 画像モデルにつかするデータ拡張ステージの定義


# In[51]:


data_augmentation=keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2)
    ]
)


# In[52]:


# ランダムに水増しされた訓練画像の表示


# In[55]:


plt.figure(figsize=(10,10))
for images, _ in train_dataset.take(1):
    for i in range(9):
        augmented_images=data_augmentation(images)
        ax=plt.subplot(3,3,i+1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis


# In[58]:


# 犬と猫を分類するためのcnnをインスタンス化(データ拡張とドロップアウトを追加した新しいCNN)
inputs=keras.Input(shape=(180,180,3))
x=data_augmentation(inputs)
x=layers.Rescaling(1./255)(inputs)
x=layers.Conv2D(filters=32, kernel_size=3, activation='relu')(inputs)
x=layers.MaxPooling2D(pool_size=2)(x)
x=layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
x=layers.MaxPooling2D(pool_size=2)(x)
x=layers.Conv2D(filters=128, kernel_size=3, activation='relu')(x)
x=layers.MaxPooling2D(pool_size=2)(x)
x=layers.Conv2D(filters=256, kernel_size=3, activation='relu')(x)
x=layers.MaxPooling2D(pool_size=2)(x)
x=layers.Conv2D(filters=256, kernel_size=3, activation='relu')(x)
x=layers.Flatten()(x)
x=layers.Dropout(0.5)(x)
outputs=layers.Dense(1, activation='sigmoid')(x)
model=keras.Model(inputs=inputs, outputs=outputs)


# In[60]:


# コンパイル


# In[61]:


model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=["accuracy"])


# In[63]:


# 正則化したcnnを訓練


# In[64]:


callbacks=[
    ModelCheckpoint(filepath="convnet_r_cat&dog.h5",
                                   save_best_only=True,
                                   monitor='val_loss',period=1)
]
history=model.fit(train_dataset, epochs=100, validation_data=validation_dataset, callbacks=[callbacks])


# In[65]:


# 訓練時の損失値と正解率をプロット(正則化後)
import matplotlib.pyplot as plt
accuracy=history.history["accuracy"]
val_accuracy=history.history["val_accuracy"]
loss=history.history["loss"]
val_loss=history.history["val_loss"]
epochs=range(1, len(accuracy)+1)
plt.figure()
plt.plot(epochs, accuracy, 'bo', label="training_accuracy")
plt.plot(epochs, val_accuracy, 'b', label="validation_accuracy")
plt.title("training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label="training_loss")
plt.plot(epochs, val_loss, 'b', label="validation_loss")
plt.title("training and validation loss")
plt.legend()
plt.show()


# In[66]:


test_model=keras.models.load_model("convnet_r_cat&dog.h5")
test_loss, test_acc=test_model.evaluate(test_dataset)
print(f"test accuracy : {test_acc:.3f}")

