#!/bin/bash

yum update gcc
yum update libstdc++
export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=1"
