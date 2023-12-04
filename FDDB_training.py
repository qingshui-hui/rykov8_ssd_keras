import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import pickle
from random import shuffle
from imageio import imread
# from scipy.misc import imresize
from PIL import Image
import tensorflow as tf
from ssd import SSD300
from ssd_training import MultiboxLoss
from ssd_utils import BBoxUtility
from PIL import Image
from lxml import etree
from xml.etree import ElementTree
import math
import os
import glob
# from google.colab.patches import cv2_imshow

os.makedirs('checkpoints', exist_ok=True)#学習途中の重みも保存するためのフォルダ作成
checkpoints_dir='checkpoints'

np.set_printoptions(suppress=True)

NUM_CLASSES = 2
input_shape = (300, 300, 3)

# 計算済みのデフォルトボックスの位置を保存したprior_boxes_ssd300を読み込む
# SSDは画像上に大きさや形の異なるデフォルトボックスを乗せ、その枠ごとに予測値を計算
# VGGやResNetのような画像分類で大きな成果をあげたネットワーク構造を用いて画像から特徴マップを抽出
# そして特徴マップの位置毎に候補を用意（SSD論文ではdefault boxと呼ばれている）
priors = pickle.load(open('prior_boxes_ssd300.pkl', 'rb'))
# バウンティボックス
bbox_util = BBoxUtility(NUM_CLASSES, priors)

# FDDBデータの情報まとめたファイル
gt = pickle.load(open('FDDB.pkl', 'rb'))
keys = sorted(gt.keys())
num_train = int(round(0.8 * len(keys)))
train_keys = keys[:num_train]
val_keys = keys[num_train:]
num_val = len(val_keys)

class Generator(object):# pkl上にpathも含めているのでpath_prefixは消した
    def __init__(self, gt, bbox_util,
                 batch_size,
                 train_keys, val_keys, image_size,
                 saturation_var=0.5,
                 brightness_var=0.5,
                 contrast_var=0.5,
                 lighting_std=0.5,
                 hflip_prob=0.5,
                 vflip_prob=0.5,
                 do_crop=True,
                 crop_area_range=[0.75, 1.0],
                 aspect_ratio_range=[3./4., 4./3.]):
        self.gt = gt
        self.bbox_util = bbox_util
        self.batch_size = batch_size
        self.train_keys = train_keys
        self.val_keys = val_keys
        self.train_batches = len(train_keys)
        self.val_batches = len(val_keys)
        self.image_size = image_size
        self.color_jitter = []
        if saturation_var:
            self.saturation_var = saturation_var
            self.color_jitter.append(self.saturation)
        if brightness_var:
            self.brightness_var = brightness_var
            self.color_jitter.append(self.brightness)
        if contrast_var:
            self.contrast_var = contrast_var
            self.color_jitter.append(self.contrast)
        self.lighting_std = lighting_std
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.do_crop = do_crop
        self.crop_area_range = crop_area_range
        self.aspect_ratio_range = aspect_ratio_range

    # 画像のチャンネル数変えたり明度変えたりして学習用の画像を増やしているのかな？？関数群
    def grayscale(self, rgb):
        return rgb.dot([0.299, 0.587, 0.114])

    def saturation(self, rgb):
        gs = self.grayscale(rgb)
        alpha = 2 * np.random.random() * self.saturation_var 
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
        return np.clip(rgb, 0, 255)

    def brightness(self, rgb):
        alpha = 2 * np.random.random() * self.brightness_var 
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha
        return np.clip(rgb, 0, 255)

    def contrast(self, rgb):
        gs = self.grayscale(rgb).mean() * np.ones_like(rgb)
        alpha = 2 * np.random.random() * self.contrast_var 
        alpha += 1 - self.contrast_var
        rgb = rgb * alpha + (1 - alpha) * gs
        return np.clip(rgb, 0, 255)

    def lighting(self, img):
        cov = np.cov(img.reshape(-1, 3) / 255.0, rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)
        noise = np.random.randn(3) * self.lighting_std
        noise = eigvec.dot(eigval * noise) * 255
        img += noise
        return np.clip(img, 0, 255)

    def horizontal_flip(self, img, y):
        if np.random.random() < self.hflip_prob:
            img = img[:, ::-1]
            y[:, [0, 2]] = 1 - y[:, [2, 0]]
        return img, y

    def vertical_flip(self, img, y):
        if np.random.random() < self.vflip_prob:
            img = img[::-1]
            y[:, [1, 3]] = 1 - y[:, [3, 1]]
        return img, y

    # 画像のアスペクト比変えたりして、それに合うようにBBOXも変えている感じ？？
    def random_sized_crop(self, img, targets):
        img_w = img.shape[1]
        img_h = img.shape[0]
        img_area = img_w * img_h
        random_scale = np.random.random()
        random_scale *= (self.crop_area_range[1] -
                         self.crop_area_range[0])
        random_scale += self.crop_area_range[0]
        target_area = random_scale * img_area
        random_ratio = np.random.random()
        random_ratio *= (self.aspect_ratio_range[1] -
                         self.aspect_ratio_range[0])
        random_ratio += self.aspect_ratio_range[0]
        w = np.round(np.sqrt(target_area * random_ratio))     
        h = np.round(np.sqrt(target_area / random_ratio))
        if np.random.random() < 0.5:
            w, h = h, w
        w = min(w, img_w)
        w_rel = w / img_w
        w = int(w)
        h = min(h, img_h)
        h_rel = h / img_h
        h = int(h)
        x = np.random.random() * (img_w - w)
        x_rel = x / img_w
        x = int(x)
        y = np.random.random() * (img_h - h)
        y_rel = y / img_h
        y = int(y)
        img = img[y:y+h, x:x+w]
        new_targets = []
        for box in targets:
            cx = 0.5 * (box[0] + box[2])
            cy = 0.5 * (box[1] + box[3])
            if (x_rel < cx < x_rel + w_rel and
                y_rel < cy < y_rel + h_rel):
                xmin = (box[0] - x_rel) / w_rel
                ymin = (box[1] - y_rel) / h_rel
                xmax = (box[2] - x_rel) / w_rel
                ymax = (box[3] - y_rel) / h_rel
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(1, xmax)
                ymax = min(1, ymax)
                box[:4] = [xmin, ymin, xmax, ymax]
                new_targets.append(box)
        new_targets = np.asarray(new_targets).reshape(-1, targets.shape[1])
        return img, new_targets

    # 上の関数群使って学習させる画像を生成する関数かな？？前処理とかも実施している
    def generate(self, train=True):
        print(tf.test.gpu_device_name())
        while True:
            # 1epochごとに画像の順番をシャッフル
            if train:# 学習の時は学習用ファイルを使う
                shuffle(self.train_keys)
                keys = self.train_keys
            else:
                shuffle(self.val_keys)
                keys = self.val_keys
            inputs = []
            targets = []
            for key in keys:            
                img_path = key
                #print(img_path)
                img = imread(img_path).astype('float32')
                y = self.gt[key].copy()
                if train and self.do_crop:
                    img, y = self.random_sized_crop(img, y)
                # img = imresize(img, self.image_size).astype('float32')
                img = np.array(Image.fromarray(img).resize(self.image_size, resample=2))
                # boxの位置は正規化されているから画像をリサイズしても
                # 教師信号としては問題ない　らしいです
                if train:
                    shuffle(self.color_jitter)
                    for jitter in self.color_jitter:
                        if len(img.shape) == 2:#グレイスケール画像の場合エラーが出るので3チャンネルの画像に変換する
                          img=cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                        img = jitter(img)
                    if self.lighting_std:
                        img = self.lighting(img)
                    if self.hflip_prob > 0:
                        img, y = self.horizontal_flip(img, y)
                    if self.vflip_prob > 0:
                        img, y = self.vertical_flip(img, y)
                # 訓練データ生成時にbbox_utilを使っているのはここだけらしい
                y = self.bbox_util.assign_boxes(y)
                inputs.append(img)                
                targets.append(y)
                # 1 iter 分終わったら返す
                if len(targets) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = np.array(targets)
                    inputs = []
                    targets = []
                    # 前処理。preprocess_input()の中ではモデルによって画像の正規化、ImageNetデータセットのRGB各チャンネルごとの平均値を引く、などの処理が行われているようです。
                    # VGG16では画像をRGBからBGRに変換し、スケーリングせずにImageNetデータセットに対して各カラーチャネルをゼロ中心にします。
                    yield preprocess_input(tmp_inp), tmp_targets

gen = Generator(gt, bbox_util, 16,
                train_keys, val_keys,
                (input_shape[0], input_shape[1]), do_crop=False)

# 学習済みモデル読み込み
model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.load_weights('weights_SSD300.hdf5', by_name=True)

# 再学習しないレイヤー
freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
          'conv2_1', 'conv2_2', 'pool2',
          'conv3_1', 'conv3_2', 'conv3_3', 'pool3']#,
#           'conv4_1', 'conv4_2', 'conv4_3', 'pool4']

# 再学習しないように設定
for L in model.layers:
    if L.name in freeze:
        L.trainable = False

def schedule(epoch, decay=0.9):
    return base_lr * decay**(epoch)

# driveの容量が危ないので5 epochずつ保存することにする
callbacks = [keras.callbacks.ModelCheckpoint(checkpoints_dir + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                             verbose=1,monitor='val_loss',save_best_only=True,mode='min',
                                             save_weights_only=True,period=5),
             keras.callbacks.LearningRateScheduler(schedule)]

base_lr = 3e-4
optim = keras.optimizers.Adam(lr=base_lr)
# neg_pos_ratio:ハードネガティブマイニング負例と正例の最大の比
model.compile(optimizer=optim,
              loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss)

nb_epoch = 100

# 学習
# ミニバッチごとに入力の前処理を行うのでfit_generatorを使う
history = model.fit_generator(gen.generate(True), gen.train_batches,
                              nb_epoch, verbose=1,
                              callbacks=callbacks,
                              validation_data=gen.generate(False),
                              nb_val_samples=gen.val_batches,
                              nb_worker=1)
model.save('FDDB_LEARN.h5')
with open('FDDB_HISTORY.dat', 'wb') as file_pi:
  pickle.dump(history.history, file_pi)
