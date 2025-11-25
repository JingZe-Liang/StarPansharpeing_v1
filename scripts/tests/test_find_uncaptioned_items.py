from pathlib import Path

# Has captions
caption_dir = "data/RemoteSAM270k/RemoteSAM-270K/captions/JPEGImages"
files = Path(caption_dir).rglob("*")
captions = set([f.stem for f in files])
print(f"Totally {len(captions)} captioned items.")


total_files = []
with open("data/RemoteSAM270k/total_samples.list.txt") as f:
    lines = f.readlines()
for l in lines:
    stem = Path(l.strip()).stem
    total_files.append(stem)
print(f"Totally {len(total_files)} items in total.")
# duplicated files
tt_files_set = set(total_files)
print(f"Duplicated files: {len(tt_files_set) - len(total_files)}")  # has duplicates


uncaptioned_items = list(set(total_files) - captions)
print(f"Totally {len(uncaptioned_items)} uncaptioned items.")

# Found them!
print("Saving uncaptioned items...")
with open("data/RemoteSAM270k/uncaptioned_items.list.txt", "w") as f:
    for item in uncaptioned_items:
        f.write(item + "\n")
