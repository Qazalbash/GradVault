# Radial Sweep Algorithm

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
./radial_sweep <path_to_image1> <window name1> ... <path_to_imageN> <window nameN>
```

For the next times, just run:

```bash
cd build
make
./radial_sweep <path_to_image1> <window name1> ... <path_to_imageN> <window nameN>
```

for default test cases,

```bash
./radial_sweep ../image/square.bmp Sqaure ../image/circle.bmp Circle
```

## Errors and Bugs

This code is written on Ubuntu 23.04.2 LTS. If you are using a different OS, you may face some errors. One common thing that could happen is that include paths for opencv may be different in your system. In that case, you need to edit the `CMakeLists.txt` file in the root directory. You can find the file [here](CMakeLists.txt). The other bug that I faced was about the GTK. I solved it by running `unset GTK_PATH`.
