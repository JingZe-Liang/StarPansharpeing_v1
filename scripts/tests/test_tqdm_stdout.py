import sys

# logger_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> [<level>{level: ^6}</level>] <level>{message}</level>"
# logger.configure(
#     handlers=[
#         dict(
#             sink=lambda msg: tqdm.write(msg, end=""),
#             format=logger_format,
#             colorize=True,
#             filter=lambda record: "tqdm" in record["extra"],
#         ),
#         dict(
#             sink=sys.stdout,
#             format=logger_format,
#             colorize=True,
#         ),
#     ]
# )
# for i in (tbar := tqdm(range(100), file=sys.stdout, desc="Test tqdm to stdout")):
#     time.sleep(0.1)
#     # tbar.write(f"Processing item {i}")
#     # print(f"Processing item {i}", file=sys.stdout)
#     logger.bind(tqdm=True).info(f"Processing item {i}")
#     logger.info("this is standard message")
import time

from loguru import logger
from tqdm import tqdm

logger.remove()
logger.add(
    lambda msg: tqdm.write(msg, end=""),
    colorize=True,
    filter=lambda record: "special" in record["extra"],
)
logger.add(sys.stderr, filter=lambda record: "special" not in record["extra"])

# Let's say log in module A without tqdm
logger.info("Initializing")

# Logs in module B with tqdm
for x in tqdm(range(10)):
    logger.bind(special=True).info("Iterating #{}", x)
    time.sleep(1)
