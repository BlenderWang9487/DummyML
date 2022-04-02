#!/bin/bash

if [[ !(-d src) ]]; then
  cd ..
fi

if [[ !(-d src) ]]; then
  echo "Please run this script in DummyML folder."
  exit 1
fi

if [ -n $1 ]; then
    BUILD_TYPE=$1;
else
    BUILD_TYPE=Release;
fi

cmake -B build -S . -DCMAKE_BUILD_TYPE=$BUILD_TYPE && \
    cmake --build build --config $BUILD_TYPE && \
    python3 -m pytest
