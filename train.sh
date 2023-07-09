#!/bin/bash
for (( a = 0; a < 5; a++ ))
do
  num1=0.25
  var1=$(echo "$a * $num1" | bc)
  make train ARGS="--mixing_proportion=${var1} --num_epochs=25"
  rm models/output/model.pth
done
