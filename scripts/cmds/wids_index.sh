# !/bin/bash

# if wids index does not work
# python -m wids.wids_index \
# create \
# "data/DFC_2020_public-13_bands-px_256-MSI-{0000..0017}.tar" \
# --o data/DCF2019.json \
# -n DCF2019_Hyperspectral_dataset \


# python src/data/wids_index.py \
# create \
# "data/WorldView2-8_bands-px_256-MSI-0000.tar" \
# --o data/WorldView2.json \


python src/data/wids_index.py \
create \
"data/miniFrance/miniFrance_labeled_unlabeled_3_bands-px_1024-MSI-{0000..0010}.tar" \
"data/miniFrance/miniFrance_test_3_bands-px_1024-MSI-{0000..0017}.tar" \
-o data/miniFrance/shardindex.json \
-fnd -fnm
