import os
from subprocess import check_call

def worker(video_f):
    video_name = video_f.split('.')[0]
    folder = os.path.join('/u/zkou2/Data/A2D/Release/pngs320H', video_name)
    if not os.path.exists(folder):
        os.mkdir(folder)
    frame_path = os.path.join('/u/zkou2/Data/A2D/Release/pngs320H', video_name, '%05d.png')
    result = check_call(['ffmpeg', '-i', os.path.join('/u/zkou2/Data/A2D/Release/clips320H', video_f), frame_path])
    print(result)

if __name__ == "__main__":
    if not os.path.exists('/u/zkou2/Data/A2D/Release/pngs320H'):
        os.mkdir('/u/zkou2/Data/A2D/Release/pngs320H')
    lst = os.listdir('/u/zkou2/Data/A2D/Release/clips320H')
    for vid in lst:
        worker(vid)