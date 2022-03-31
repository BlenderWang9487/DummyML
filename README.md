# DummyML
A Dummy ML Library For NSD Course Term Project (: P)

## Introduction
**DummyML** is an easy-to-use machine learning hybrid library.

The core of ML algorithm is implement in C++, and the API is provided in Python 3.

## Dependencies

DummyML has the following dependencies:

    cmake   >=  3.4
    python  >=  3
    numpy   >=  1.22.3
## Usage

    $ git clone --recursive https://github.com/BlenderWang9487/DummyML.git
    $ cd DummyML/
    $ mkdir build
    $ cd build/
    $ cmake -DCMAKE_BUILD_TYPE=Release ..
    $ make
    $ cd ..
    $ python3 -m pytest tests/