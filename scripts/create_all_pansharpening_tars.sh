#!/bin/bash

# Create tar archives for all pansharpening datasets
# This script processes all satellite datasets and creates wids indices

echo "Starting to create tar archives for all datasets..."

# IKONOS Dataset
echo "Processing IKONOS dataset..."
cd "/HardDisk/ZiHanCao/datasets/Pansharpening-NBU_RS_Data/Sat_Dataset/Dataset/1 IKONOS/1 IKONOS/MS_256"
find . -name "*.mat" | sort -V | tar cf IKONOS_MS.tar -T -
cd "/HardDisk/ZiHanCao/datasets/Pansharpening-NBU_RS_Data/Sat_Dataset/Dataset/1 IKONOS/1 IKONOS/PAN_1024"
find . -name "*.mat" | sort -V | tar cf IKONOS_PAN.tar -T -

# QuickBird Dataset
echo "Processing QuickBird dataset..."
cd "/HardDisk/ZiHanCao/datasets/Pansharpening-NBU_RS_Data/Sat_Dataset/Dataset/2 QuickBird/2 QuickBird/MS_256"
find . -name "*.mat" | sort -V | tar cf QuickBird_MS.tar -T -
cd "/HardDisk/ZiHanCao/datasets/Pansharpening-NBU_RS_Data/Sat_Dataset/Dataset/2 QuickBird/2 QuickBird/PAN_1024"
find . -name "*.mat" | sort -V | tar cf QuickBird_PAN.tar -T -

# Gaofen-1 Dataset
echo "Processing Gaofen-1 dataset..."
cd "/HardDisk/ZiHanCao/datasets/Pansharpening-NBU_RS_Data/Sat_Dataset/Dataset/3 Gaofen-1/3 Gaofen-1/MS_256"
find . -name "*.mat" | sort -V | tar cf Gaofen1_MS.tar -T -
cd "/HardDisk/ZiHanCao/datasets/Pansharpening-NBU_RS_Data/Sat_Dataset/Dataset/3 Gaofen-1/3 Gaofen-1/PAN_1024"
find . -name "*.mat" | sort -V | tar cf Gaofen1_PAN.tar -T -

# WorldView-4 Dataset
echo "Processing WorldView-4 dataset..."
cd "/HardDisk/ZiHanCao/datasets/Pansharpening-NBU_RS_Data/Sat_Dataset/Dataset/4 WorldView-4/4 WorldView-4/MS_256"
find . -name "*.mat" | sort -V | tar cf WorldView4_MS.tar -T -
cd "/HardDisk/ZiHanCao/datasets/Pansharpening-NBU_RS_Data/Sat_Dataset/Dataset/4 WorldView-4/4 WorldView-4/PAN_1024"
find . -name "*.mat" | sort -V | tar cf WorldView4_PAN.tar -T -

# WorldView-2 Dataset
echo "Processing WorldView-2 dataset..."
cd "/HardDisk/ZiHanCao/datasets/Pansharpening-NBU_RS_Data/Sat_Dataset/Dataset/5 WorldView-2/5 WorldView-2/MS_256"
find . -name "*.mat" | sort -V | tar cf WorldView2_MS.tar -T -
cd "/HardDisk/ZiHanCao/datasets/Pansharpening-NBU_RS_Data/Sat_Dataset/Dataset/5 WorldView-2/5 WorldView-2/PAN_1024"
find . -name "*.mat" | sort -V | tar cf WorldView2_PAN.tar -T -

# WorldView-3 Dataset
echo "Processing WorldView-3 dataset..."
cd "/HardDisk/ZiHanCao/datasets/Pansharpening-NBU_RS_Data/Sat_Dataset/Dataset/6 WorldView-3/6 WorldView-3/MS_256"
find . -name "*.mat" | sort -V | tar cf WorldView3_MS.tar -T -
cd "/HardDisk/ZiHanCao/datasets/Pansharpening-NBU_RS_Data/Sat_Dataset/Dataset/6 WorldView-3/6 WorldView-3/PAN_1024"
find . -name "*.mat" | sort -V | tar cf WorldView3_PAN.tar -T -

echo "All tar archives created successfully!"

# Move tar files to target directory and create wids indices
echo "Moving files and creating wids indices..."
cd "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/data"

# Create directories for each dataset
mkdir -p "IKONOS/pansharpening_full"
mkdir -p "QuickBird/pansharpening_full"
mkdir -p "Gaofen1/pansharpening_full"
mkdir -p "WorldView4/pansharpening_full"
mkdir -p "WorldView2/pansharpening_full"
mkdir -p "WorldView3/pansharpening_full"

# Move IKONOS files
mv "/HardDisk/ZiHanCao/datasets/Pansharpening-NBU_RS_Data/Sat_Dataset/Dataset/1 IKONOS/1 IKONOS/MS_256/IKONOS_MS.tar" "IKONOS/pansharpening_full/"
mv "/HardDisk/ZiHanCao/datasets/Pansharpening-NBU_RS_Data/Sat_Dataset/Dataset/1 IKONOS/1 IKONOS/PAN_1024/IKONOS_PAN.tar" "IKONOS/pansharpening_full/"

# Move QuickBird files
mv "/HardDisk/ZiHanCao/datasets/Pansharpening-NBU_RS_Data/Sat_Dataset/Dataset/2 QuickBird/2 QuickBird/MS_256/QuickBird_MS.tar" "QuickBird/pansharpening_full/"
mv "/HardDisk/ZiHanCao/datasets/Pansharpening-NBU_RS_Data/Sat_Dataset/Dataset/2 QuickBird/2 QuickBird/PAN_1024/QuickBird_PAN.tar" "QuickBird/pansharpening_full/"

# Move Gaofen-1 files
mv "/HardDisk/ZiHanCao/datasets/Pansharpening-NBU_RS_Data/Sat_Dataset/Dataset/3 Gaofen-1/3 Gaofen-1/MS_256/Gaofen1_MS.tar" "Gaofen1/pansharpening_full/"
mv "/HardDisk/ZiHanCao/datasets/Pansharpening-NBU_RS_Data/Sat_Dataset/Dataset/3 Gaofen-1/3 Gaofen-1/PAN_1024/Gaofen1_PAN.tar" "Gaofen1/pansharpening_full/"

# Move WorldView-4 files
mv "/HardDisk/ZiHanCao/datasets/Pansharpening-NBU_RS_Data/Sat_Dataset/Dataset/4 WorldView-4/4 WorldView-4/MS_256/WorldView4_MS.tar" "WorldView4/pansharpening_full/"
mv "/HardDisk/ZiHanCao/datasets/Pansharpening-NBU_RS_Data/Sat_Dataset/Dataset/4 WorldView-4/4 WorldView-4/PAN_1024/WorldView4_PAN.tar" "WorldView4/pansharpening_full/"

# Move WorldView-2 files
mv "/HardDisk/ZiHanCao/datasets/Pansharpening-NBU_RS_Data/Sat_Dataset/Dataset/5 WorldView-2/5 WorldView-2/MS_256/WorldView2_MS.tar" "WorldView2/pansharpening_full/"
mv "/HardDisk/ZiHanCao/datasets/Pansharpening-NBU_RS_Data/Sat_Dataset/Dataset/5 WorldView-2/5 WorldView-2/PAN_1024/WorldView2_PAN.tar" "WorldView2/pansharpening_full/"

# Move WorldView-3 files
mv "/HardDisk/ZiHanCao/datasets/Pansharpening-NBU_RS_Data/Sat_Dataset/Dataset/6 WorldView-3/6 WorldView-3/MS_256/WorldView3_MS.tar" "WorldView3/pansharpening_full/"
mv "/HardDisk/ZiHanCao/datasets/Pansharpening-NBU_RS_Data/Sat_Dataset/Dataset/6 WorldView-3/6 WorldView-3/PAN_1024/WorldView3_PAN.tar" "WorldView3/pansharpening_full/"

echo "All files moved successfully!"

# Create wids indices for all datasets
echo "Creating wids indices..."

# IKONOS indices
wids-index create "IKONOS/pansharpening_full/IKONOS_MS.tar" -o "IKONOS/pansharpening_full/MS_shardindex.json"
wids-index create "IKONOS/pansharpening_full/IKONOS_PAN.tar" -o "IKONOS/pansharpening_full/PAN_shardindex.json"

# QuickBird indices
wids-index create "QuickBird/pansharpening_full/QuickBird_MS.tar" -o "QuickBird/pansharpening_full/MS_shardindex.json"
wids-index create "QuickBird/pansharpening_full/QuickBird_PAN.tar" -o "QuickBird/pansharpening_full/PAN_shardindex.json"

# Gaofen-1 indices
wids-index create "Gaofen1/pansharpening_full/Gaofen1_MS.tar" -o "Gaofen1/pansharpening_full/MS_shardindex.json"
wids-index create "Gaofen1/pansharpening_full/Gaofen1_PAN.tar" -o "Gaofen1/pansharpening_full/PAN_shardindex.json"

# WorldView-4 indices
wids-index create "WorldView4/pansharpening_full/WorldView4_MS.tar" -o "WorldView4/pansharpening_full/MS_shardindex.json"
wids-index create "WorldView4/pansharpening_full/WorldView4_PAN.tar" -o "WorldView4/pansharpening_full/PAN_shardindex.json"

# WorldView-2 indices
wids-index create "WorldView2/pansharpening_full/WorldView2_MS.tar" -o "WorldView2/pansharpening_full/MS_shardindex.json"
wids-index create "WorldView2/pansharpening_full/WorldView2_PAN.tar" -o "WorldView2/pansharpening_full/PAN_shardindex.json"

# WorldView-3 indices
wids-index create "WorldView3/pansharpening_full/WorldView3_MS.tar" -o "WorldView3/pansharpening_full/MS_shardindex.json"
wids-index create "WorldView3/pansharpening_full/WorldView3_PAN.tar" -o "WorldView3/pansharpening_full/PAN_shardindex.json"

echo "All wids indices created successfully!"
echo "Script completed!"
