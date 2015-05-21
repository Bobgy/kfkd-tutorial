#!/bin/bash
source $LASAGNE_ENV
python -u model/$1/kfkd.py plot_image $2
