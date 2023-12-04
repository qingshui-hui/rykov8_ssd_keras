import cv2
from lxml import etree
import math
import os
import glob

os.makedirs('Annotations', exist_ok=True)#xmlを保存するフォルダを作っておく
save_dir='Annotations'
def writeXML(imName, faces, H, W, C):

    annotation = etree.Element('annotation')

    folder = etree.SubElement(annotation, 'folder').text='VOC2007'
    filename = etree.SubElement(annotation, 'filename').text=imName+'.jpg'
    source = etree.SubElement(annotation, 'source')

    database = etree.SubElement(source, 'database').text='The FDDB Database'
    annno = etree.SubElement(source, 'annotation').text='FDDB'
    image = etree.SubElement(source, 'image').text='Dummy'
    flickrid = etree.SubElement(source, 'flickrid').text='Dummy'

    owner = etree.SubElement(annotation, 'owner')
    flickrid2 = etree.SubElement(owner, 'flickrid').text='Dummy'
    name = etree.SubElement(owner, 'name').text='Dummy'

    size = etree.SubElement(annotation, 'size')
    width = etree.SubElement(size, 'width').text=str(W)
    height = etree.SubElement(size, 'height').text=str(H)
    depth = etree.SubElement(size, 'depth').text=str(C)

    segmented = etree.SubElement(annotation, 'segmented').text='0'

    for face in faces:
        obj = etree.SubElement(annotation, 'object')
        name2 = etree.SubElement(obj, 'name').text='face'
        pose = etree.SubElement(obj, 'pose').text='Unspecified'
        truncated = etree.SubElement(obj, 'truncated').text='0'
        difficult = etree.SubElement(obj, 'difficult').text='0'

        bndbox = etree.SubElement(obj, 'bndbox')

        xmin = etree.SubElement(bndbox, 'xmin').text=str(face[0])
        ymin = etree.SubElement(bndbox, 'ymin').text=str(face[1])
        xmax = etree.SubElement(bndbox, 'xmax').text=str(face[2])
        ymax = etree.SubElement(bndbox, 'ymax').text=str(face[3])

    tree = etree.ElementTree(annotation)
    tree.write(save_dir+'/'+imName.split('/')[-1]+".xml",pretty_print=True)



anno_dir='FDDB-folds'
pic_dir='originalPics'

# アノテーションtxtファイルファイルリストの取得
file_list=[os.path.split(f)[1] for f in glob.glob(anno_dir+'/*ellipseList.txt')]
print(file_list)
print('file Cnt: ', len(file_list))

for file in file_list:
  print(file)
  f = open(anno_dir+'/'+file)
  '''
  #txtファイルの中身
  2002/08/31/big/img_18008　ファイル名
  4　顔の数
  53.968100 38.000000 -1.494904 31.598276 55.596600  1　座標とか角度1
  56.000000 37.000000 -1.460399 202.152999 122.034200  1　座標とか角度2
  54.558400 39.000000 1.396263 293.611040 133.853600  1　座標とか角度3
  44.000000 34.000000 -1.442216 391.131100 168.266900  1　座標とか角度4　ここで1ファイル分
  2002/08/22/big/img_249　ファイル名
  1　顔の数
  92.731568 55.547794 1.319755 133.877336 101.823201  1　座標とか角度1
  '''
  while True:
    line = f.readline()#実行するたびに1行ずつ読み取っていく。最初はファイル名
    if not line:
        break

    line = line.strip()
    imName = pic_dir+'/'+line

    # idx = line.rfind('/')
    # imName = line[idx+1:]

    print('processing ' + imName)

    # '2002/08/11/big/img_591'
    im = cv2.imread(imName+'.jpg')

    H, W, C = im.shape
    faceNum = int(f.readline().strip())#実行するたびに1行ずつ読み取っていく。これは1ファイルの情報行数
    faces = []

    for faceIdx in range(faceNum):
        #実行するたびに1行ずつ読み取っていく。faceNum行分実行。これは座標とか角度
        line = f.readline().strip()
        splited = line.split()
        r1 = float(splited[0])
        r2 = float(splited[1])
        angle = float(splited[2])
        cx = float(splited[3])
        cy = float(splited[4])


        rectH = 2*r1*(math.cos(math.radians(abs(angle))))
        rectW = 2*r2*(math.cos(math.radians(abs(angle))))

        lx = int(max(0, cx - rectW/2))
        ly = int(max(0, cy - rectH/2))
        rx = int(min(W-1, cx + rectW/2))
        ry = int(min(H-1, cy + rectH/2))

        faceIdx = 0

        faces.append((lx,ly,rx,ry))

    writeXML(imName, faces, H, W, C)

  f.close()