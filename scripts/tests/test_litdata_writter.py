import numpy as np
from litdata.streaming.writer import BinaryWriter

writter = BinaryWriter("tmp/bins/", chunk_size=10)
print("Writing data to tmp/bins/")

for i in range(100):
    writter.add_item(i, {"data": np.random.randn(1, 3, 3), "label": i % 10})

writter.done()
writter.merge()
print("Done writing data")
