# data/scripts/preprocess.sh

#!/bin/bash

python data/scripts/preprocess.py \
    --data_dir ../data \
    --tokenizer_name t5-small \
    --max_source_length 128 \
    --max_target_length 128
