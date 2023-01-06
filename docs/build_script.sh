#!/usr/bash

rm -rf _local_build build generated
#sphinx-autogen -o generated source/reference/*.rst && sphinx-build ./source _local_build &&
make docs
