
from utils.NoteSet import T_S,T_F


# 选择乐器
Musical_Instruments = {
    'Acoustic_Grand_Piano':0,
    'Bright_Acoustic_Piano':1,
    'Electric_Grand_Piano':2,
    'Celesta':8,
    'Glockenspiel':9,
    'Music_box':10,
    'Hammond_Organ':16,
    'Violin':40,
    'Bassoon':70,
    'Woodblock':115,
    'Tango_Accordian':23,
    'Seashore':122,
    'Applause':126,
    'Gunshot':127,
}

# 决定谱号
Determine_clef = {
    'Gclef':0,
    'High_Gclef':1,
    'DHigh_Gclef':2,
    'Lower_Gclef':3,
    'DLower_Gclef':4,
    'Fclef':5,
    'High_Fclef':6,
    'DHigh_Fclef':7,
    'Lower_Fclef':8,
    'DLower_Fclef':9,
    'Cclef':10,
    'Soprano_Cclef':11,
    'M_Soprano_Cclef':12,
    'Tensor_Cclef':13,
    'Baritone_Cclef':14,}

# 真实时长
Actual_Note_Time = {
    1:8,  2:4,  4:2,  8:1,  16:0.5, 32:0.25}

# 真实Gcelf音高
Actual_Gclef_High = {-10:47,
    -9:48, -8:50, -7:52, -6:53, -5:55, -4:57, -3:59,
    -2:60, -1:62, 0:64,  1:65,  2:67,  3:69,  4:71,
    5:72,  6:74,  7:76,  8:77,  9:79,  10:81, 11:83,
    12:84, 13:86, 14:88, 15:89, 16:91, 17:93, 18:95,
    19:96, 20:98, }

# 真实高八度Gcelf音高
Actual_High_Gclef_High = {-10:59,
    -9:60, -8:62, -7:64,  -6:65,  -5:67,  -4:69,  -3:71,
    -2:72, -1:74, 0:76,   1:77,   2:79,   3:81,   4:83,
    5:84,  6:86,  7:88,   8:89,   9:91,   10:93,  11:95,
    12:96, 13:98, 14:100, 15:101, 16:103, 17:105, 18:107,
    19:108}
# 真实倍八度Gcelf音高
Actual_DHigh_Gclef_High = {-10:71,
    -9:72, -8:74, -7:76, -6:77, -5:79, -4:81,  -3:83,
    -2:84, -1:86, 0:88,  1:89,  2:91,  3:93,   4:95,
    5:96,  6:98,  7:100, 8:101, 9:103, 10:105, 11:107,
    12:108,}
# 真实低八度Gcelf音高
Actual_Lower_Gclef_High = {-10:35,
    -9:36, -8:38, -7:40, -6:41, -5:43, -4:45, -3:47,
    -2:48, -1:50, 0:52,  1:53,  2:55,  3:57,  4:59,
    5:60,  6:62,  7:64,  8:65,  9:67,  10:69, 11:71,
    12:72, 13:74, 14:76, 15:77, 16:79, 17:81, 18:83,
    19:84, 20:86, }
# 真实倍低八度Gcelf音高
Actual_DLower_Gclef_High = {-10:23,
    -9:24, -8:26, -7:28, -6:29, -5:31, -4:33, -3:35,
    -2:36, -1:38, 0:40,  1:41,  2:43,  3:45,  4:47,
    5:48,  6:50,  7:52,  8:53,  9:55,  10:57, 11:59,
    12:60, 13:62, 14:64, 15:65, 16:67, 17:69, 18:71,
    19:72, 20:74, }

# 真实Fcelf音高
Actual_Fclef_High = {
    -10:26,-9:28, -8:29, -7:31, -6:33, -5:35,
    -4:36, -3:38, -2:40, -1:41, 0:43,  1:45,  2:47,
    3:48,  4:50,  5:52,  6:53,  7:55,  8:57,  9:59,
    10:60, 11:62, 12:64, 13:65, 14:67, 15:69, 16:71,
    17:72, 18:74, 19:76, 20:77, }
# 真实高八度Fcelf音高
Actual_High_Fclef_High = {
    -10:38, -9:40, -8:41, -7:43, -6:45, -5:47,
    -4:48,  -3:50, -2:52, -1:53, 0:55,  1:57,  2:59,
    3:60,   4:62,  5:64,  6:65,  7:67,  8:69,  9:71,
    10:72,  11:74, 12:76, 13:77, 14:79, 15:81, 16:83,
    17:84,  18:86, 19:88, 20:89, }
# 真实倍高八度Fcelf音高
Actual_DHigh_Fclef_High = {
    -10:50, -9:52, -8:53,  -7:55, -6:57, -5:59,
    -4:60,  -3:62, -2:64,  -1:65, 0:67,  1:69,  2:71,
    3:72,   4:74,  5:76,   6:77,  7:79,  8:81,  9:83,
    10:84,  11:86, 12:88,  13:89, 14:91, 15:93, 16:95,
    17:96,  18:98, 19:100, 20:101, }
# 真实低八度Fcelf音高
Actual_Lower_Fclef_High = {
    -4:24, -3:26, -2:28, -1:29, 0:31,  1:33,  2:35,
    3:36,  4:38,  5:40,  6:41,  7:43,  8:45,  9:47,
    10:48, 11:50, 12:52, 13:53, 14:55, 15:57, 16:59,
    17:60, 18:62, 19:64, 20:65, }
# 真实倍低八度Fcelf音高
Actual_DLower_Fclef_High = {
    3:24,  4:26,  5:28,  6:29,  7:31,  8:33,  9:35,
    10:36, 11:38, 12:40, 13:41, 14:43, 15:45, 16:47,
    17:48, 18:50, 19:52, 20:53, }

# 真实Ccelf音高
Actual_Cclef_High = {
    -10:36, -9:38, -8:40, -7:41, -6:43, -5:45, -4:47,
    -3:48,  -2:50, -1:52, 0:53,  1:55,  2:57,  3:59,
    4:60,   5:62,  6:64,  7:65,  8:67,  9:69,  10:71,
    11:72,  12:74, 13:76, 14:77, 15:79, 16:81, 17:83,
    18:84,  19:86, 20:88}
# 真实女高音Ccelf音高
Actual_Soprano_Cclef_High = {-10:43, -9:45, -8:47,
    -7:48,  -6:50, -5:52, -4:53, -3:55, -2:57, -1:59,
    0:60,   1:62,  2:64,  3:65,  4:67,  5:69,  6:71,
    7:72,   8:74,  9:76,  10:77, 11:79, 12:81, 13:83,
    14:84,  15:86, 16:88, 17:89, 18:91, 19:93, 20:95, }
# 真实女中音Ccelf音高
Actual_M_Soprano_Cclef_High = {
    -10:40, -9:41, -8:43, -7:45, -6:47,
    -5:48,  -4:50, -3:52, -2:53, -1:55, 0:57, 1:59,
    2:60,   3:62,  4:64,  5:65,  6:67,  7:69,  8:71,
    9:72,   10:74, 11:76, 12:77, 13:79, 14:81, 15:83,
    16:84,  17:86, 18:88, 19:89, 20:91, }
# 真实次中音Ccelf音高
Actual_Tensor_Cclef_High = {-10:33,-9:35,
    -8:36, -7:38, -6:40, -5:41, -4:43, -3:45, -2:47,
    -1:48, 0:50,  1:52,  2:53,  3:55,  4:57,  5:59,
    6:60,  7:62,  8:64,  9:65,  10:67, 11:69, 12:71,
    13:72, 14:74, 15:76, 16:77, 17:79, 18:81, 19:83,
    20:84, }
# 真实男中音Ccelf音高
Actual_Baritone_Cclef_High = {-10:29,-9:31,-8:33,-7:35,
    -6:36, -5:38, -4:40, -3:41, -2:43, -1:45, 0:47,
    1:48,  2:50,  3:52,  4:53,  5:55,  6:57,  7:59,
    8:60,  9:62,  10:64, 11:65, 12:67, 13:69, 14:71,
    15:72, 16:74, 17:76, 18:77, 19:79, 20:81, }

Clef_High = [
    Actual_Gclef_High,
    Actual_High_Gclef_High,
    Actual_DHigh_Gclef_High,
    Actual_Lower_Gclef_High,
    Actual_DLower_Gclef_High,
    Actual_Fclef_High,
    Actual_High_Fclef_High,
    Actual_DHigh_Fclef_High,
    Actual_Lower_Fclef_High,
    Actual_DLower_Fclef_High,
    Actual_Cclef_High,
    Actual_Soprano_Cclef_High,
    Actual_M_Soprano_Cclef_High,
    Actual_Tensor_Cclef_High,
    Actual_Baritone_Cclef_High,]


# 继续扩充

Determine_T = {
    'Gclef':{"Sharp":[8],"D_S":[8,5],"A_S":[8,5,9],"E_S":[8,5,9,6],"B_S":[8,5,9,6,3],"F_S":[8,5,9,6,3,7],"C_S":[8,5,9,6,3,7,4],
             "Flat":[4],"B_F":[4,7],"E_F":[4,7,3],"A_F":[4,7,3,6],"D_F":[4,7,3,6,2],"G_F":[4,7,3,6,2,5],"C_F":[4,7,3,6,2,5,1]},
    'Fclef':{"Sharp":[6],"D_S":[6,3],"A_S":[6,3,7],"E_S":[6,3,7,4],"B_S":[6,3,7,4,1],"F_S":[6,3,7,4,1,5],"C_S":[6,3,7,4,1,5,2],
             "Flat":[2],"B_F":[2,5],"E_F":[2,5,1],"A_F":[2,5,1,4],"D_F":[2,5,1,4,0],"G_F":[2,5,1,4,0,3],"C_F":[2,5,1,4,0,3,-1]},
    'Cclef':{"Sharp":[7],"D_S":[7,4],"A_S":[7,4,8],"E_S":[7,4,8,5],"B_S":[7,4,8,5,2],"F_S":[7,4,8,5,2,6],"C_S":[7,4,8,5,2,6,3],
             "Flat":[3],"B_F":[3,6],"E_F":[3,6,2],"A_F":[3,6,2,5],"D_F":[3,6,2,5,1],"G_F":[3,6,2,5,1,4],"C_F":[3,6,2,5,1,4,0]},
}


def detT(clef, T):
    num = []
    try:
        for t in Determine_T[clef][T]:
            while t <= 20:
                num.append(t)
                t += 7
            while -10 <= t:
                t -= 7
                if t >= -10:
                    num.append(t)
        num = list(set(num))
        return num
    except:
        return []

def detL(C,T,L,pitch):
    if L == "Sharp":
        return 1
    elif L == "Flat":
        return -1
    else:
        if pitch in detT(C,T):
            if T in T_S:
                return -1
            elif T in T_F:
                return 1
            else:
                pass
        else:
            return 0


if __name__ == '__main__':
    print(detT('Gclef', 'Flat'))