@echo off
gcc -shared -O2 -o ..\..\bin\ufbx.dll ufbx.c ufbx_bridge.c -lm
echo Built bin\ufbx.dll