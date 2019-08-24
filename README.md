# BirdEyeView
How to build the project :

Here is the CMake command for you:
```bash
$ cd ~
$ git clone https://github.com/fabriziodiblasi/BirdEyeView.git Progetto_HPC
$ cd Progetto_HPC
$ mkdir Build
$ cd Build
$ cmake -D CMAKE_C_COMPILER=/usr/bin/gcc-6 ..
$ make 
```

## Install dependencies

```bash
sudo apt-get install git
sudo apt-get install build-essential
sudo apt-get install cmake
sudo apt-get install pkg-config
sudo apt-get install libgtkglext1 libgtkglext1-dev
sudo apt-get install yasm
```
### CUDA

```bash
sudo ubuntu-drivers devices
sudo ubuntu-drivers devices autoinstall
sudo apt-get install nvidia-cuda-toolkit
```

### FFMPEG
```bash
git clone git://source.ffmpeg.org/ffmpeg.git ffmpeg

cd ffmpeg

./configure --enable-nonfree --enable-pic --enable-shared --enable-avresample

make -j4

sudo make install
```

### OPENCV
```bash

cd ~/<my_working_directory>

wget -O opencv.zip https://github.com/opencv/opencv/archive/4.1.1.zip 

unzip opencv.zip -d .

git clone https://github.com/opencv/opencv_contrib.git

cd ~/opencv-4.1.1

mkdir build

cd build

cmake -D CMAKE_C_COMPILER=/usr/bin/gcc-6 -D WITH_OPENCL=ON -D WITH_OPENGL=ON -D WITH_CUDA=ON -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda -DWITH_NVCUVID=OFF -DBUILD_opencv_cudacodec=ON -D WITH_V4L=ON -D WITH_FFMPEG=ON -D WITH_GSTREAMER=ON  -D WITH_FFMPEG=ON -D OPENCV_EXTRA_MODULES_PATH=<my_working_directory>/opencv_contrib/modules ..

make -j4 (depending on number of cores -> nproc)

sudo make install
```

### Controllo del corretto funzionamento di FFMPEG 

```bash
eseguire i seguenti comandi :

[root@host ~]# ffmpeg
ffmpeg: error while loading shared libraries: libavdevice.so.56: cannot open shared object file: No such file or directory

This error usually caused if the libraries are not loaded. 
Try below command to see the required libraries by the ffmpeg command.


[root@host ~]# ldd $(which ffmpeg)
        linux-vdso.so.1 =>  (0x00006efe6b90f000)
        libavdevice.so.56 => not found
        libavfilter.so.5 => not found
        libavformat.so.56 => not found
        libavcodec.so.56 => not found
        libpostproc.so.53 => not found
        libswresample.so.1 => not found
        libswscale.so.3 => not found
        libavutil.so.54 => not found
        libm.so.6 => /lib64/libm.so.6 (0x00006efe6b67a000)
        libpthread.so.0 => /lib64/libpthread.so.0 (0x00006efe6b45d000)
        libc.so.6 => /lib64/libc.so.6 (0x00006efe6b0c8000)
        /lib64/ld-linux-x86-64.so.2 (0x00006efe6b910000)

You will see “not found” message even the libraries are present on server. So to fix it, 
you will need to find out the path of the libraries using below commands. 

        
[root@host ~]# updatedb && locate libavdevice.so
        /opt/ffmpeg/libavdevice/libavdevice.so
        /opt/ffmpeg/libavdevice/libavdevice.so.56
        /usr/local/lib/libavdevice.so
        /usr/local/lib/libavdevice.so.56
        /usr/local/lib/libavdevice.so.56.4.100

On my server, the path of the library is /usr/local/lib/. 
Find out the path on your server and append it in the file /etc/ld.so.conf the simply run below command to load the libraries. 

[root@host ~]# ldconfig

Now check the required libraries for ffmpeg and you should see: 

[root@host ~]# ldd $(which ffmpeg)
        linux-vdso.so.1 =>  (0x000068716560a000)
        libavdevice.so.56 => /usr/local/lib/libavdevice.so.56 (0x00006871653e7000)
        libavfilter.so.5 => /usr/local/lib/libavfilter.so.5 (0x000068716505b000)
        libavformat.so.56 => /usr/local/lib/libavformat.so.56 (0x0000687164c77000)
        libavcodec.so.56 => /usr/local/lib/libavcodec.so.56 (0x00006871636bb000)
        libpostproc.so.53 => /usr/local/lib/libpostproc.so.53 (0x0000687163473000)
        libswresample.so.1 => /usr/local/lib/libswresample.so.1 (0x0000687163257000)
        libswscale.so.3 => /usr/local/lib/libswscale.so.3 (0x0000687162fc6000)
        libavutil.so.54 => /usr/local/lib/libavutil.so.54 (0x0000687162d5f000)
        libm.so.6 => /lib64/libm.so.6 (0x0000687162ada000)
        libpthread.so.0 => /lib64/libpthread.so.0 (0x00006871628bd000)
        libc.so.6 => /lib64/libc.so.6 (0x0000687162529000)
        libxcb.so.1 => /usr/lib64/libxcb.so.1 (0x000068716230a000)
        libxcb-shm.so.0 => /usr/lib64/libxcb-shm.so.0 (0x0000687162108000)
        libxcb-xfixes.so.0 => /usr/lib64/libxcb-xfixes.so.0 (0x0000687161f02000)
        libxcb-shape.so.0 => /usr/lib64/libxcb-shape.so.0 (0x0000687161cfe000)
        libasound.so.2 => /lib64/libasound.so.2 (0x0000687161a13000)
        libz.so.1 => /lib64/libz.so.1 (0x00006871617fd000)
        libx264.so.148 => /usr/local/lib/libx264.so.148 (0x0000687161443000)
        libvpx.so.1 => /usr/lib64/libvpx.so.1 (0x00006871610b8000)
        libvorbisenc.so.2 => /usr/lib64/libvorbisenc.so.2 (0x0000687160cde000)
        libvorbis.so.0 => /usr/lib64/libvorbis.so.0 (0x0000687160ab1000)
        libtheoraenc.so.1 => /usr/lib64/libtheoraenc.so.1 (0x0000687160884000)
        libtheoradec.so.1 => /usr/lib64/libtheoradec.so.1 (0x0000687160675000)
        libopencore-amrwb.so.0 => /usr/lib64/libopencore-amrwb.so.0 (0x000068716045e000)
        libopencore-amrnb.so.0 => /usr/lib64/libopencore-amrnb.so.0 (0x000068716022f000)
        libmp3lame.so.0 => /usr/lib64/libmp3lame.so.0 (0x000068715ffb8000)
        libfaac.so.0 => /usr/lib64/libfaac.so.0 (0x000068715fda5000)
        librt.so.1 => /lib64/librt.so.1 (0x000068715fb9d000)
        /lib64/ld-linux-x86-64.so.2 (0x000068716560b000)
        libXau.so.6 => /usr/lib64/libXau.so.6 (0x000068715f999000)
        libdl.so.2 => /lib64/libdl.so.2 (0x000068715f795000)
        libogg.so.0 => /usr/lib64/libogg.so.0 (0x000068715f58e000)
        libstdc++.so.6 => /usr/lib64/libstdc++.so.6 (0x000068715f288000)
        libgcc_s.so.1 => /lib64/libgcc_s.so.1 (0x000068715f072000)

```


