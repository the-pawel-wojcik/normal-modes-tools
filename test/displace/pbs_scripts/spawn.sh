#!/bin/bash

set -e 

split -l 15 --numeric-suffixes displacements.xyz displaced

for fname in displaced*
do
  newdir=${fname:9}
  echo $newdir
  if [[ -d $newdir ]]
  then
    rm -fr $newdir
  fi
  mkdir $newdir
  mv $fname $newdir
  cd $newdir
  cp ../ZMAT ./ZMATtmp
  tail -14 $fname > tmp
  cat tmp ZMATtmp > ZMAT
  rm tmp ZMATtmp $fname
  cd ..
done
