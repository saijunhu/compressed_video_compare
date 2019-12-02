import os

ROOT = '/home/sjhu/datasets/UCF-101'
class Dataset:
    def __init__(self,path):
        self.root_path = path
        
    def preprocess(self):
        for folder in os.listdir(self.root_path):
            for video in os.listdir(os.path.join(self.root_path, folder)):
                os.chdir(os.path.join(self.root_path, folder))
                video_folder = video.split('.')[0]
                os.mkdir(video_folder)
                os.system("mv %s %s" %(video, video_folder))

    def process(self):
        for folder in os.listdir(self.root_path):
            for videodir in os.listdir(os.path.join(self.root_path, folder)):
                self.generate_varient(os.path.join(self.root_path, folder,videodir))

    def generate_varient(self,videodir):
        os.chdir(videodir)
        origin_avi = os.listdir(videodir)[0]

        #case 1: avi to mp4
        origin_mp4 = origin_avi.strip().split('.')[0]+'.mp4'
        return_code_a = os.system(
                    "/home/sjhu/env/ffmpeg-4.2-amd64-static/ffmpeg -loglevel warning  -i %s -vf scale=340:256,setsar=1:1 -c:v libx264 %s" % (
                        origin_avi, origin_mp4))

        #case 2: scale 1/2
        origin_scaled = origin_avi.strip().split('.')[0] + '_scaled.mp4'
        return_code_a = os.system(
                    "/home/sjhu/env/ffmpeg-4.2-amd64-static/ffmpeg  -loglevel warning  -i %s -vf scale=170:128,setsar=1:1 -c:v libx264 %s" % (
                        origin_avi, origin_scaled))

        #case 3: crop center
        origin_center_crop = origin_avi.strip().split('.')[0] + '_center_crop.mp4'
        return_code_a = os.system(
                    "/home/sjhu/env/ffmpeg-4.2-amd64-static/ffmpeg  -loglevel warning  -i %s -vf crop=iw*3/4:ih*3/4:iw/8:ih/8,setsar=1:1 -c:v libx264 %s" % (
                        origin_avi, origin_center_crop))

        #case 4: video quality set
        origin_low_quality = origin_avi.strip().split('.')[0] + '_low_quality.mp4'
        return_code_a = os.system(
                    "/home/sjhu/env/ffmpeg-4.2-amd64-static/ffmpeg -loglevel warning  -i %s -vf scale=340:256,setsar=1:1 -c:v libx264 -crf 40 %s" % (
                        origin_avi, origin_low_quality))
        # case 5 : video quality set
        origin_high_quality = origin_avi.strip().split('.')[0] + '_high_quality.mp4'
        return_code_a = os.system(
            "/home/sjhu/env/ffmpeg-4.2-amd64-static/ffmpeg -loglevel warning  -i %s -vf scale=340:256,setsar=1:1  -c:v libx264 -crf 18 %s" % (
                origin_avi, origin_high_quality))


if __name__ == '__main__':
    d = Dataset(ROOT)
    d.process()
