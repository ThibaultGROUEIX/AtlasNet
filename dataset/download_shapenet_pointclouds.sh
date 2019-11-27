#!/bin/bash
function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}
cd dataset
mkdir data
gdrive_download 1OG-IxPSJok3in78uSAX26GiIJUa-kt2Y data/ShapeNetV1PointCloud.zip
unzip data/ShapeNetV1PointCloud.zip
cd ..
