#!usr/bin/bash


if  ! [[ -d ./train_imgs/ && -d ./test_imgs/ &&  -d ./saved_model/ ]]; then
	mkdir "train_imgs"
	mkdir "test_imgs"
	mkdir "saved_model"
fi

python3 pix2pix.py $1

