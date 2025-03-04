#!/bin/bash

for fname in ??
do
  grep 'Total EOMEE-CCSD energy:' $fname/output.c4
done
