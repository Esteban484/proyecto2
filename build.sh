#!/usr/bin/env bash
#exit on error
set -o errexit
pip install updgrade pip
pip install flask
pip install opencv-python
pip install joblib=="1.2.0"
pip install sklearn
