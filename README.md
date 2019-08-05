# BirdEyeView

## Install dependencies

### FFMPEG

git clone git://source.ffmpeg.org/ffmpeg.git ffmpeg

cd ffmpeg

./configure --enable-nonfree --enable-pic --enable-shared --enable-avresample

make -j4

sudo make install

### OPENCV

cd ~/<my_working_directory>

git clone https://github.com/opencv/opencv.git

git clone https://github.com/opencv/opencv_contrib.git

cd ~/opencv

mkdir build

cd build

cmake  -D WITH_V4L=ON -D WITH_FFMPEG=ON -D WITH_GSTREAMER=ON  -D WITH_FFMPEG=ON -D OPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules <opencv_source_directory> 

make -j4 (depending on number of cores -> nproc)

make install


