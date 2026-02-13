GraphCut_RANSAC
===============

Build instructions
------------------

Required tools:

- CMake
- Git
- C/C++ compiler (GCC, Visual Studio or Clang)

Required libraries:

- OpenCV with modules

Note:

- CMAKE variables you can configure:

  - USE_OPENMP (ON(default)/OFF)
      - Parallelize using OpenMP
	  
Compiling
---------

```shell
$ git clone https://github.com/danini/magsac
$ cmake -S . -B build
$ cmake --build build
```

- CMake: Configure + Generate
- CMake: Set the OpenCV_DIR if needed.
- Build
