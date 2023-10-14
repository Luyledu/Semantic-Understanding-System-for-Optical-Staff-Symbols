import xml.etree.cElementTree as Et
import os
from PIL import Image,ImageDraw,ImageFont
import torch
import matplotlib

def drawRect(img, pos, **kwargs):
    transp = Image.new('RGBA', img.size, (0,0,0,0))
    draw = ImageDraw.Draw(transp, "RGBA")
    draw.rectangle(pos, **kwargs)
    img.paste(Image.alpha_composite(img, transp))

def read_xml_bbox(xml_filepath):
    xml_information = []
    tree = Et.parse(xml_filepath)
    root = tree.getroot()
    for obj in root.iter('object'):
        cls = obj.find('name').text
        xmlbox = obj.find('bndbox')
        cxxyy = (cls, float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
        xml_information.append(cxxyy)
    return xml_information

def read_txt_bbox(txt_filepath):
    txt_information = []
    with open(txt_filepath,'r') as f:
        for line in f.readlines():
            c,x1,y1,x2,y2 = line.strip().split(' ')
            txt_information.append([c,x1,x2,y1,y2])
    return txt_information


def main(img_path,xml_path,txt_path,save_path,font_size,xml=True):
    filenames = os.listdir(img_path)
    if '.DS_Store' in filenames:
        filenames.remove('.DS_Store')
    for filename in filenames:
        img_filepath = os.path.join(img_path,filename)
        xml_filepath = os.path.join(xml_path,filename.replace('jpg','xml'))
        txt_filepath = os.path.join(txt_path,filename.replace('jpg','txt'))
        save_filepath = os.path.join(save_path,filename.replace('jpg','pdf'))
        if xml:
            information = read_xml_bbox(xml_filepath)
        else:
            information = read_txt_bbox(txt_filepath)
        img = Image.open(img_filepath).convert('RGBA')
        draw = ImageDraw.Draw(img)
        for bbox in information:
            c,x1,x2,y1,y2 = bbox[0],int(bbox[1]),int(bbox[2]),int(bbox[3]),int(bbox[4])
            if 'error' in c:
                drawRect(img, (x1, y1, x2, y2), fill=(90, 180, 40, 100))
                c = c[:-6]
            else:
                drawRect(img, (x1, y1, x2, y2), fill=(0, 110, 220, 100))
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Times New Roman.ttf", font_size)
            draw.text((x1,y1-(font_size + 5)),c,(0,0,0),font=font)
        rgb = Image.new('RGB', img.size, (255, 255, 255))
        rgb.paste(img, mask=img.split()[3])
        rgb.save(save_filepath)

if __name__ == '__main__':
    xml_path = "/Users/loufengbin/Documents/python/pythonProject/tensorflow/YOLO/yolov5-6.1/runs/detect/2009_detect/2_0/xml"
    txt_path = "/Users/loufengbin/Documents/python/pythonProject/tensorflow/YOLO/yolov5-6.1/runs/val/exp30/labels"
    img_path = "/Users/loufengbin/Documents/python/pythonProject/tensorflow/YOLO/yolov5-6.1/test2009/2"
    save_path = "/Users/loufengbin/Documents/python/pythonProject/tensorflow/YOLO/yolov5-6.1/runs/detect/2009_detect/2_2"
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    main(img_path,xml_path,txt_path,save_path,30,xml=True)

