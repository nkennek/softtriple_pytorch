#!/bin/sh


if [ -z "$1" ]
  then
    workdir=$(dirname $0)/../
    cd $workdir
else
    workdir=$1
    cd $workdir
fi

curl -O http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
tar -zxvf CUB_200_2011.tgz -C $workdir && rm CUB_200_2011.tgz
