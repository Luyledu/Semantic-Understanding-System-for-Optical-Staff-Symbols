import os

from PIL import Image, ImageDraw, ImageFont,ImageOps
from utils.score import NotationParsing,ScorePretreatmest

def drawRect(img, pos, **kwargs):
    transp = Image.new('RGBA', img.size, (0,0,0,0))
    draw = ImageDraw.Draw(transp, "RGBA")
    draw.rectangle(pos, **kwargs)
    img.paste(Image.alpha_composite(img, transp))

def drewNoteToImg(img_path,mung_path,output_path):
    img_filenames = os.listdir(img_path)
    if '.DS_Store' in img_filenames:
        img_filenames.remove('.DS_Store')
    for img_filename in img_filenames:
        img_filename_path = os.path.join(img_path,img_filename)
        mung_filename = img_filename.replace('jpg','csv')
        mung_filename_path = os.path.join(mung_path,mung_filename)
        save_filepath = os.path.join(output_path,img_filename.replace("jpg","pdf"))

        MN_E = NotationParsing(ScorePretreatmest(mung_filename_path))
        score_img = Image.open(img_filename_path).convert('RGBA')
        score_img = ImageOps.exif_transpose(score_img)
        draw = ImageDraw.Draw(score_img)
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Times New Roman.ttf", 30)
        for line in MN_E:
            for NOTE in line:
                F = 0
                name,coo = NOTE.split('#')
                try:
                    xmin,ymin,xmax,ymax = coo.split('/')
                except:
                    xmin, ymin, xmax, ymax, F = coo.split('/')
                # draw.rectangle((int(xmin), int(ymin), int(xmax), int(ymax)), fill=None, outline='blue', width=3)
                # draw.text((int(xmin), int(ymin) - 30), name, fill='blue', font=font)
                if F:
                    drawRect(score_img, (int(xmin), int(ymin), int(xmax), int(ymax)), fill=(90, 80, 240, 100))
                else:
                    drawRect(score_img, (int(xmin), int(ymin), int(xmax), int(ymax)), fill=(252, 125, 61, 100))
                draw.text((int(xmin), int(ymin) - 33), name, (0, 0, 0), font=font)
        rgb = Image.new('RGB', score_img.size, (255, 255, 255))
        rgb.paste(score_img, mask=score_img.split()[3])
        rgb.save(save_filepath)
        # if not os.path.isdir(output_path):
        #     os.mkdir(output_path)
        # score_img.save(output_path + f'/{img_filename.replace("jpg","pdf")}')
        print(f"{img_filename}已保存")

if __name__ == '__main__':
    img_path = '/Users/loufengbin/Documents/python/pythonProject/tensorflow/YOLO/yolov5-6.1/test2009/2'
    mung_path = '/Users/loufengbin/Documents/python/pythonProject/tensorflow/YOLO/yolov5-6.1/runs/detect/2009_detect/2_0/csv'
    output_path = '/Users/loufengbin/Documents/python/pythonProject/tensorflow/YOLO/yolov5-6.1/runs/detect/2009_detect/2_3'
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    drewNoteToImg(img_path,mung_path,output_path)