 
# Direct Fourier Transform - CUDA Implementation
###### Note: currently only supported by NVIDIA GPUs (limitation of CUDA), generalized OpenCL version in progress.
---
##### Instructions for installation of this software (includes profiling, linting, building, and unit testing):
1. Ensure you have an NVIDIA based GPU (**mandatory!**)
2. Install the [CUDA](https://developer.nvidia.com/cuda-downloads) toolkit and runtime (refer to link for download/installation procedure)
3. Install [Valgrind](http://valgrind.org/) (profiling, memory checks, memory leaks etc.)
   ```bash
   $ sudo apt install valgrind
   ```
4. Install [Cmake](https://cmake.org/)/[Makefile](https://www.gnu.org/software/make/) (build tools)
   ```bash
   $ sudo apt install cmake
   ```
5. Install [Google Test](https://github.com/google/googletest) (unit testing) - See [this tutorial](https://www.eriksmistad.no/getting-started-with-google-test-on-ubuntu/) for tutorial on using Google Test library
   ```bash
   $ sudo apt install libgtest-dev
   $ cd /usr/src/gtest
   $ sudo cmake CMakeLists.txt
   $ sudo make
   $ sudo cp *.a /usr/lib
   ```
6. Install [Cppcheck](http://cppcheck.sourceforge.net/) (linting)
   ```bash
   $ sudo apt install cppcheck
   ```
7. Configure the code for usage (**modify direct_fourier_transform.cu config**)
8. Create local execution folder
    ```bash
   $ mkdir build && cd build
   ```
9. Build direct fourier transform project (from project folder)
   ```bash
   $ cmake .. -DCMAKE_BUILD_TYPE=Release && make
   ```
10. **Important: set -CDMAKE_BUILD_TYPE=Debug if planning to run Valgrind. Debug mode disables compiler optimizations, which is required for Valgrind to perform an optimal analysis.**
---
##### Instructions for usage of this software (includes executing, testing, linting, and profiling):
To perform memory checking, memory leak analysis, and profiling using [Valgrind](http://valgrind.org/docs/manual/quick-start.html), execute the following (assumes you are in the appropriate *build* folder (see step 5 above):
```bash
$ valgrind --leak-check=yes -v ./dft
$ valgrind --leak-check=yes -v ./tests
```
To execute linting, execute the following commands (assumes you are in the appropriate source code folder):
```bash
$ cppcheck --enable=all main.cpp
$ cppcheck --enable=all direct_fourier_transform.cu
$ cppcheck --enable=all unit_testing.cpp
```
To execute unit testing, execute the following (also assumes appropriate *build* folder):
```bash
$ ./tests
````
To execute the direct fourier transform (once configured and built), execute the following command (also assumes appropriate *build* folder):
```bash
$ ./dft
```