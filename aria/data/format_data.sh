#!/bin/bash

set -e

START_SEQ=$1
END_SEQ=$2
ROOT_DIR=$3
ARIA_TOOLS=$4
PYTHON="/usr/local/bin/python3.11"


MAX_JOBS=10
running_jobs=0

mkdir -p "tmp_data"

for ((i=$START_SEQ; i<$END_SEQ; i+=10)); do
	$PYTHON "$ARIA_TOOLS"/projects/AriaSyntheticEnvironment/aria_synthetic_environments_downloader.py \
		--set train \
		--scene-ids "$i-$((i+9))" \
		--cdn-file "$ROOT_DIR"/aria_synthetic.json \
		--output-dir "$ROOT_DIR"/tmp_data \
		--unzip True \

	for ((j=i; j<i+10; j+=1)); do
		
		(		
		$PYTHON $ROOT_DIR/generate_pairs.py \
			--scene $j \
			--root "$ROOT_DIR" \
		) &
			
		echo $j

	done
	wait
		
	rm -rf "$ROOT_DIR"/tmp_data/*
	
	echo 'Finished chunk'
done

echo 'Finished all chunks'


