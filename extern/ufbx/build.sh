#!/bin/bash
if [[ "$OSTYPE" == "darwin"* ]]; then
    gcc -shared -fPIC -O2 -o ../../bin/ufbx.dylib ufbx.c ufbx_bridge.c -lm
    echo "Built bin/ufbx.dylib"
else
    gcc -shared -fPIC -O2 -o ../../bin/ufbx.so ufbx.c ufbx_bridge.c -lm
    echo "Built bin/ufbx.so"
fi