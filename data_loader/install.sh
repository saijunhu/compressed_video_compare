#!/usr/bin/env bash
rm -rf build
# compile coviar_data_loader.c ,generate a module in bulid/,
python setup.py build_ext --inplace
# install the module
python setup.py install --user

