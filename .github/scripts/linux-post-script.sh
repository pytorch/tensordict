#!/bin/bash

if [ "$(uname)" != "Darwin" ]; then
  yum update gcc
  yum update libstdc++
else
  brew update
  brew upgrade gcc
fi
