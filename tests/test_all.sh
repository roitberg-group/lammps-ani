#!/bin/bash
set -ex

cd ./test_ani2x_nocuaev_double
./test.sh
cd run
./run.sh
cd ../../

