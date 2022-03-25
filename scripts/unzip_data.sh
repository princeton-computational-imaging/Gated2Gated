#!/bin/bash

download_root=$1
dst=$2

files=(
	$download_root/gated2gated.z01
	$download_root/gated2gated.z02
	$download_root/gated2gated.z03
	$download_root/gated2gated.z04
	$download_root/gated2gated.z05
	$download_root/gated2gated.z06
	$download_root/gated2gated.z07
	$download_root/gated2gated.z08
	$download_root/gated2gated.z09
	$download_root/gated2gated.z10
	$download_root/gated2gated.z11
	$download_root/gated2gated.zip
	)
mkdir -p $dst
all_exists=true
for item in ${files[*]} 
do
	if [[ ! -f "$item" ]]; then
    		echo "$item is missing"
		all_exists=false
	fi
done

if $all_exists; then
	zip -s- $download_root/gated2gated.zip -O $dst/gated2gated_full.zip
	unzip $dst/gated2gated_full.zip
fi



