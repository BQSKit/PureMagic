#!/usr/bin/bash

egrep -v "//|^$|include|OPEN" $1|wc -l
