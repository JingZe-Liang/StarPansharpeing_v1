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
"data/MUSLI-438_bands-px_512-MSI-0000.tar" \
"data/MUSLI-438_bands-px_512-MSI-0001.tar" \
"data/MUSLI-438_bands-px_512-MSI-0002.tar" \
"data/MUSLI-438_bands-px_512-MSI-0003.tar" \
"data/MUSLI-438_bands-px_512-MSI-0004.tar" \
"data/MUSLI-438_bands-px_512-MSI-0005.tar" \
"data/MUSLI-438_bands-px_512-MSI-0006.tar" \
"data/MUSLI-438_bands-px_512-MSI-0007.tar" \
"data/MUSLI-438_bands-px_512-MSI-0008.tar" \
"data/MUSLI-438_bands-px_512-MSI-0009.tar" \
--o data/MUSLI.json \
-fnd  # do not copy into /tmp/shard.tar file for the local tar files
