git clone https://github.com/gflags/gflags.git
cd gflags
git checkout v2.2.2

mkdir build_
cd build_

cmake .. -G Ninja
ninja