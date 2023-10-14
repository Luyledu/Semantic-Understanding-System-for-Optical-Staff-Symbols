import os

from utils.mido_yin import Determine_clef,Actual_Note_Time,Clef_High
from utils.mido_yin import Determine_T,detT,detL
from utils.NoteSet import L,C,T,S,R,T_S,T_F
import fractions
import csv
import time


classes = ['DSharp','TimeSig','Rests1', 'Rests2', 'Rests4', 'Rests8', 'Rests16', 'Rests32', 'Rests64', 'Rests128',
           'Bass', 'Guit', 'Tpt', 'Hn', 'Tbn', 'Tba', 'Fl', 'Ob', 'Cl', 'Bsn', 'Vln', 'Vla', 'Vlc', 'Pno', 'Wh', 'Trb',
           'Sax', 'Hrp', 'Cb', 'Rec', 'Euph', 'Timp', 'Picc', 'Kl', 'Vc', 'Dbs', 'Hch', 'Cel', 'Cbs',
           's', 'm', 'p', 'f', 'mp', 'mf', 'sf', 'ff', 'fff', 'ffff', 'fffff', 'pp', 'ppp', 'pppp', 'ppppp', 'fp','sfz', 'rfz', 'sfp',
           'Speed_4=80', 'Speed_4=160', 'Speed_4=123', 'Speed_2=80', 'Speed_4=95', 'Speed_4=57', 'Speed_4=123','Speed_4=50',
           'Speed_4=120', 'Speed_4=100', 'Speed_4=110', 'Speed_4=105','Speed_4=150', 'Speed_4=140','Speed_4=84', 'Speed_4=60','Speed_4=30',]




def ScorePretreatmest(csv_path):
    score = []

    with open(csv_path, encoding='utf-8-sig') as f:
        for line in csv.reader(f, skipinitialspace=True):
            line.pop(0)

            score_line = {}
            Note_num = 0
            for noteAndCoo in line[::-1]:
                note = noteAndCoo.split('#')[0]
                if note in R:
                    line.remove(noteAndCoo)
                if note == 'TimeSig':
                    line.remove(noteAndCoo)
                if note in S:
                    Note_num += 1

            score_line["C"] = 0
            score_line["T"] = 0
            for i in range(1,Note_num+2):
                score_line[i] = {"p":0,"d":0,"v1":0,"v2":0,"coo":0}
            j = 0
            n = 0
            N_j = len(line)
            score_line['C'] = line[0].split('#')[0]
            if line[1].split('#')[0] in T:
                score_line['T'] = line[1].split('#')[0]
                j = 1
            else:
                score_line['T'] = 0

            while j <= (N_j - 2):
                j+=1
                if "#" in line[j]:
                    NOTE,coo = line[j].split("#")
                else:
                    NOTE = line[j]

                if NOTE in L:
                    score_line[n+1]["v1"] = NOTE

                else:
                    if NOTE in S:
                        n+=1
                        PandD = NOTE.split("_")
                        d = PandD[0]
                        p = PandD[1:]
                        score_line[n]["p"] = p
                        score_line[n]["d"] = d
                        score_line[n]["coo"] = coo
                        score_line[n]["v2"] = 0
                        score_line[n+1]["v1"] = 0
                    if NOTE == 'dot':

                        if line[j-1].split('#')[0] in S:
                        # if line[j-1] in S:
                            score_line[n]["v2"] = NOTE
            score_line.pop(Note_num+1)
            print(score_line)
            score.append(score_line)

    return score

def ScorePretreatmest_0(line):
    score = []
    score_line = {}
    Note_num = 0
    for noteAndCoo in line[::-1]:
        note = noteAndCoo.split('#')[0]
        if note in R:
            line.remove(noteAndCoo)
        if note == 'TimeSig':
            line.remove(noteAndCoo)
        if note in S:
            Note_num += 1

    score_line["C"] = 0
    score_line["T"] = 0
    for i in range(1,Note_num+2):
        score_line[i] = {"p":0,"d":0,"v1":0,"v2":0,"coo":0}
    j = 0
    n = 0
    N_j = len(line)
    score_line['C'] = line[0].split('#')[0]
    if line[1].split('#')[0] in T:
        score_line['T'] = line[1].split('#')[0]
        j = 1
    else:
        score_line['T'] = 0

    while j <= (N_j - 2):
        j+=1
        if "#" in line[j]:
            NOTE,coo = line[j].split("#")
        else:
            NOTE = line[j]

        if NOTE in L:
            score_line[n+1]["v1"] = NOTE
        else:
            if NOTE in S:
                n+=1
                PandD = NOTE.split("_")
                d = PandD[0]
                p = PandD[1:]
                score_line[n]["p"] = p
                score_line[n]["d"] = d
                # score_line[n]["coo"] = coo
                score_line[n]["v2"] = 0
                score_line[n+1]["v1"] = 0
            if NOTE == 'dot':
                if line[j - 1] in S:
                    score_line[n]["v2"] = NOTE
    score_line.pop(Note_num+1)
    score.append(score_line)
    print(score)
    return score



def NotationParsing(score):
    MN_E = []
    for line in score:
        MN_E_m = []
        C = line["C"]
        T = line["T"]

        # 考虑谱号
        clef = Clef_High[Determine_clef[C]]
        clef_new = clef.copy()

        if T:
            num = detT(C,T)
            # 考虑调号
            if T in T_S:
                for t in num:
                    clef_new[t] = clef_new[t] + 1
            elif T in T_F:
                for t in num:
                    clef_new[t] = clef_new[t] - 1

        N_n = len(line) - 2
        for n in range(1,N_n + 1):
            encode_note = ''
            p, d, v1, v2, coo = line[n]['p'], 1/int(line[n]['d']), line[n]['v1'], line[n]['v2'], line[n]['coo']
            for pitch in p:
                ActualPitch = clef_new[int(pitch)]
                # 考虑升降
                if v1:
                    ActualPitch = ActualPitch + detL(C=C,T=T,L=v1,pitch=int(pitch))
                encode_note += str(ActualPitch)
            if v2:
                d = d * (1 + 1/2)
            encode_note = str(fractions.Fraction(d)) + "_" + encode_note + "#" + str(coo)
            # encode_note = str(fractions.Fraction(d)) + "_" + encode_note
            MN_E_m.append(encode_note)
        MN_E.append(MN_E_m)
    return MN_E

def readTXT(txt_path):
    list_score = []
    f = open(txt_path,'r')
    ia = 0
    for line in f.readlines():
        ia += 1
        print(ia)
        line = line.strip('\n').split()

        for i in line[::-1]:
            if i[0:5] == 'Rests':
                line.remove(i)

        score = ScorePretreatmest_0(line)
        MN_E = NotationParsing(score)
        for i in MN_E:
            print(f"音符个数{len(i)}")
        print(MN_E)

def readCSV(csv_path,save_path):
    list_score = []
    with open(csv_path, encoding='utf-8-sig') as f:
        for row in csv.reader(f, skipinitialspace=True):
            row.pop(0)
            for note in row[::-1]:
                if note in classes:
                    row.remove(note)


            score = ScorePretreatmest_0(row)
            MN_E = NotationParsing(score)
            list_score.append(MN_E)

    list11 = []
    for line in range(len(list_score)):
        _list = list_score[line]
        for i in _list:
            i.insert(0, str(int((line / 2) + 1)))
            list11.append(i)

    with open(save_path, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerows(list11)




if __name__ == '__main__':
    mung_path = '/Users/loufengbin/Documents/python/pythonProject/tensorflow/YOLO/yolov5-6.1/runs/detect/2009_detect/9_0/csv'
    save_path = '/Users/loufengbin/Documents/python/pythonProject/tensorflow/YOLO/yolov5-6.1/runs/detect/2009_detect/9_2'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    mung_filenames = os.listdir(mung_path)
    for m in mung_filenames:
        print(m)
        csv_path = os.path.join(mung_path,m)
        save_csv_path = os.path.join(save_path,m)
        readCSV(csv_path,save_csv_path)






    # score = ScorePretreatmest(mung_path)
    # MN_E = NotationParsing(score)
    # print(MN_E)

    # csv_path = '/Users/loufengbin/Documents/music/score/000202.csv'
    # save_path = '/Users/loufengbin/Documents/music/score/bbb.csv'
    # list = readCSV(mung_path,save_path)

    # tem = ["Sharp",'Flat','Natural']
    # list = readTXT(mung_path)

    # list=[
    #     'Gclef', 'A_F', '16_5', '16_7', '16_10', '16_7', '16_5', '16_10', '16_7', 'Natural', '16_6', '16_7', '16_8',
    #     '16_7', '16_4', '16_7', '16_9', '16_7', '16_4', '16_9', '16_4', '16_7', '16_9', '16_7', '16_4', '16_9', '16_5',
    #     '16_7', '16_10', '16_7', '16_5', '16_7', '16_3', '16_4', '16_3', '16_4', '16_3', '16_2', 'Natural', '16_0',
    #     '16_-1', '16_0', '16_-1', '16_0', '16_2', '16_0', '16_-1', '16_-3', '16_-1', '16_0'
    # ]
    # score = ScorePretreatmest_0(list)
    # MN_E = NotationParsing(score)
    # a=0
    # for j in list:
    #     if j in tem:
    #         a+=1
    # print(a)
    # for i in MN_E:
    #     print(len(i))
    # print(MN_E)




