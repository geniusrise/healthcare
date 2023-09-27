#!/usr/bin/env bash

set -o errexit

git clone https://github.com/facebookresearch/faiss.git
cd faiss

rm -rf build || true

CMAKE_ROOT=$(pwd)/../venv cmake -B build . \
    -DFAISS_ENABLE_GPU=ON \
    -DFAISS_ENABLE_PYTHON=ON \
    -DFAISS_ENABLE_RAFT=OFF \
    -DBUILD_TESTING=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DFAISS_ENABLE_C_API=ON \
    -DCMAKE_BUILD_TYPE=Release

make -C build -j swigfaiss

cd build/faiss/python && python setup.py install
cd ../../..
sudo make -C build install
