#!/bin/bash

if [ "$(uname)" != "Darwin" ]; then
  yum update gcc
  yum update libstdc++
fi
