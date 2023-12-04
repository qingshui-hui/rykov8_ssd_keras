#%%
import cv2
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import pickle
from imageio import imread
from ssd import SSD300
from ssd_utils import BBoxUtility

input_shape = (300, 300, 3)
NUM_CLASSES = 2
checkpoints_dir='checkpoints'

model = SSD300(input_shape, num_classes=NUM_CLASSES)#NUM_CLASSES=2
model.load_weights('FDDB_LEARN.h5', by_name=True)#19epochの重み

priors = pickle.load(open('prior_boxes_ssd300.pkl', 'rb'))
bbox_util = BBoxUtility(NUM_CLASSES, priors)

# 画像とBBOXを可視化する関数
def plot_img_bbox(img_path):
  inputs = []
  images = []
  img = image.load_img(img_path, target_size=(300, 300))
  img = image.img_to_array(img)
  images.append(imread(img_path))
  inputs.append(img.copy())
  # 前処理
  inputs = preprocess_input(np.array(inputs))

  preds = model.predict(inputs, batch_size=1, verbose=1)
  results = bbox_util.detection_out(preds)

  for i, img2 in enumerate(images):
    img = img2.copy()
    if len(img.shape) == 2:
      img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # Parse the outputs.
    det_label = results[i][:, 0]
    det_conf = results[i][:, 1]
    det_xmin = results[i][:, 2]
    det_ymin = results[i][:, 3]
    det_xmax = results[i][:, 4]
    det_ymax = results[i][:, 5]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    colors = plt.cm.hsv(np.linspace(0, 1, NUM_CLASSES)).tolist()

    plt.imshow(img / 255.)
    currentAxis = plt.gca()

    for i in range(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * img.shape[1]))
        ymin = int(round(top_ymin[i] * img.shape[0]))
        xmax = int(round(top_xmax[i] * img.shape[1]))
        ymax = int(round(top_ymax[i] * img.shape[0]))
        score = top_conf[i]
        label = int(top_label_indices[i])
        # label_name = voc_classes[label - 1]
        display_txt = '{:0.2f}, {}'.format(score, label)
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        color = colors[label]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})

    plt.show()

img_path = 'pics/315222_s.jpg'
plot_img_bbox(img_path)
