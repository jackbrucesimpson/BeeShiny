# BeeShiny

## Description

Program written to identify tags on the back of honeybees, identify the tag type using machine learning, track hundreds of bees simultaneously, and write the output to a CSV file.

## To Do

- [x] Add background averaging to program
- [ ] Tidy up code based (code a little messy due to experimentation)

## Compiling

```
make -f Makefile.mac
make -f Makefile.linux
```

`beeshiny` executable will be created in the `bin` directory.

## Requirements

* Caffe deep learning library
* OpenCV
* C++11
* Boost

## Caffe

The `caffe` directory contains the scripts and prototxt files I used to train my caffe model that I load into the main program to recognise tags. The output model from this training, classes file and architecture file are stored in the `data` directory.

## Linking Caffe

Compiling my own C++ code and linking to Caffe has been a bit of a hassle, here's how I managed to do it after seeking advice and a lot of Googling. Hopefully this will serve as a reminder for me in the future and help anyone else with who might come across my repisitory. Keep in mind that I've only used Caffe in CPU mode.

1. At the top of your .cpp program which is linking to the Caffe library, you will need to make the following definition:
`#define CPU_ONLY`

2. Now if we try to compile anything now, Caffe will make this complaint: `caffe/proto/caffe.pb.h: No such file or directory` - some of the header files are missing from the Caffe include directory. Thus, you'll need to generate them with these commands from within the Caffe root directory:
`protoc src/caffe/proto/caffe.proto --cpp_out=.`
`mkdir include/caffe/proto`
`mv src/caffe/proto/caffe.pb.h include/caffe/proto`

3. Finally, I  copied libcaffe.so into `/usr/lib` and the caffe directory containing the header libraries (`$caffe_root/include/caffe`) into the `/usr/include directory`. To compile this on a Mac (after installing OpenBLAS with Homebrew), I just had to run:
`g++ classification.cpp -lcaffe -lglog -lopencv_core -lopencv_highgui -lopencv_imgproc -I /usr/local/Cellar/openblas/0.2.14_1/include -L /usr/local/Cellar/openblas/0.2.14_1/lib  -o classifier`

4. Alternatively, you could do what I did on my Linux machine and instead of copying header files, I just linked directly to those directories when I compiled:
`g++ classification.cpp  -lcaffe -lglog -lopencv_core -lopencv_highgui -lopencv_imgproc -I ~/caffe/include -L ~/caffe/build/lib -I /usr/local/Cellar/openblas/0.2.14_1/include -L /usr/local/Cellar/openblas/0.2.14_1/lib -o classifier`
