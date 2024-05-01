# Bayer's Demosaicing Algorithm

## Dependencies

- CMake>=3.25
- g++>=12.3.0
- OpenCV>=4.6.0

## How to compile

For the first time run the following commands:

```bash
mkdir build
cd build
cmake ..
make
./demosaic <input-image-path> <output-image-path>
```

For the next times, just run:

```bash
cd build
make
./demosaic <input-image-path> <output-image-path>
```

for default test cases,

```bash
./demosaic ../image/mandi.tif ../image/mq06861.tif
```

## Errors and Bugs

This code is written on Ubuntu 22.04.2 LTS. If you are using a different OS, you may face some errors. One common thing that could happen is that include paths for opencv may be different in your system. In that case, you need to edit the `CMakeLists.txt` file in the root directory. You can find the file [here](CMakeLists.txt). The other bug that I faced was about the GTK. I solved it by running `unset GTK_PATH`.
