#!/bin/bash

set -e

for fname in ??
do
  cd $fname
  cfourpbs huge SrOPh$fname
  cd ..
done
