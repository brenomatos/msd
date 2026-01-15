import matplotlib.pyplot as plt
import json
import os
import urllib
import yt_dlp
from tqdm import tqdm
import pandas as pd

with open("dump.json", "r") as f:
    data = json.load(f)
    

## getting set of urls; we'll use this to download the respective videos
urls_list = []
for page in data:
    for item in page:
        url = item["origem_links"]
        if("/channel/" not in url): #we exlude channels; using a channel url prompts yt-dlp to download all videos from the respective channel
          urls_list.append(url)
        else:
          print(url)
        
urls_list = set(urls_list)

# this option downloads only the audio
# ydl_opts = {
#     'keepvideo': False,
#     'format': 'bestaudio/best',
#     'postprocessors': [{
#         'key': 'FFmpegExtractAudio',
#         'preferredcodec': 'mp3',
#         'preferredquality': '192'    }],
#     'outtmpl': "%(id)s.%(ext)s"
# }

# this option downloads the videos in mp4 format
ydl_opts = {
    'keepvideo': True,
    'format': 'mp4/bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp4'}],
    'outtmpl': "%(id)s.%(ext)s"
}

videos = urls_list

for v in tqdm(videos):
  with yt_dlp.YoutubeDL(ydl_opts) as ydl:
      try:
        ydl.download(v)
      except Exception as e:
        with open("log_videos.txt","a") as f:
          f.write(str(e)+"\n")
