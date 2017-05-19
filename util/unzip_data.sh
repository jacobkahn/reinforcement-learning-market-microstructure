#!/bin/bash

# must end in a /
SOURCE_PATH=/mnt/pollux-new/cis/kearnsgroup/kearnsgroup/RLtrade/data-input-zipped/

# use wildcard. only the last extension suffix will be truncated
FILE_EXTENSION_TO_UNZIP=*.csv.gz

# must end in a /
OUTPUT_PATH=/mnt/pollux-new/cis/kearnsgroup/kearnsgroup/RLtrade/data-output-unzipped/

for f in $SOURCE_PATH$FILE_EXTENSION_TO_UNZIP
do
	filename="${f##*/}"
	echo "Unzipping $filename"
	# Unzip
	gunzip -c "$SOURCE_PATH$filename" > "$OUTPUT_PATH${filename%.*}"
done

