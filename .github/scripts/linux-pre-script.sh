#!/bin/bash

yum update gcc
yum update libstdc++

conda install conda-forge::pybind11 -y
