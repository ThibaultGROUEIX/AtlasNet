#!/bin/bash
function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}
cd dataset
gdrive_download 1JjEjv2m1BKhClOeK45MiLVH3LTe6jgjW data/ShapeNetV1Renderings.zip
cd data
unzip ShapeNetV1Renderings.zip
rm ShapeNetV1Renderings.zip
cd ..
cd ..
