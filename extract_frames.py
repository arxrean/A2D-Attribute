import os
from subprocess import check_call

def worker(video_f):
    video_name = video_f.split('.')[0]
    folder = os.path.join('/mnt/lustre/jiangsu/dlar/home/zyk17/data/A2D/Release/pngs320H', video_name)
    if not os.path.exists(folder):
        os.mkdir(folder)
    frame_path = os.path.join('/mnt/lustre/jiangsu/dlar/home/zyk17/data/A2D/Release/pngs320H', video_name, '%05d.png')
    result = check_call(['ffmpeg', '-i', os.path.join('/mnt/lustre/jiangsu/dlar/home/zyk17/data/A2D/Release/clips320H', video_f), frame_path])
    print(result)

if __name__ == "__main__":
    if not os.path.exists('/mnt/lustre/jiangsu/dlar/home/zyk17/data/A2D/Release/pngs320H'):
        os.mkdir('/mnt/lustre/jiangsu/dlar/home/zyk17/data/A2D/Release/pngs320H')
    lst = os.listdir('/mnt/lustre/jiangsu/dlar/home/zyk17/data/A2D/Release/clips320H')
    for vid in lst:
        worker(vid)