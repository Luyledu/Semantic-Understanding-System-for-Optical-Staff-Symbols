import os
import sys

import pygame
from mido import MidiFile

from utils.score import Make_score
from track import make_track


def Write_MID(score,meta_time):
    mid = MidiFile()
    track1 = make_track(score,meta_time)
    mid.tracks.append(track1)
    mid.save("music.mid")
    print(mid)
    cur_dir = sys.argv[0]
    path = os.path.join(os.path.dirname(cur_dir), "music.mid")
    return path

def play_midi(music_mung_path,meta_time,musical = 'Acoustic_Grand_Piano'):
    score = Make_score(music_mung_path,musical)
    file = Write_MID(score,meta_time)
    freq = 48000
    bitsize = -16
    channels = 2
    buffer = 1024
    pygame.mixer.init(freq, bitsize, channels, buffer)
    pygame.mixer.music.set_volume(1)
    clock = pygame.time.Clock()
    try:
        pygame.mixer.music.load(file)
    except:
        import traceback
        print(traceback.format_exc())
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        clock.tick(30)


if __name__ == '__main__':
    mung_path = '/Users/loufengbin/Documents/python/pythonProject/tensorflow/YOLO/yolov5-6.1/runs/detect/exp4/mung/000213.txt'
    meta_time = 60 * 60 * 10 / 75
    play_midi(mung_path,meta_time)
    os._exit(0)
