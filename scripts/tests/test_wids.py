import webdataset as wds
import wids
import uuid


def _add_random_id(sample):
    sample["random_id"] = str(uuid.uuid4())
    return sample


def make_loader(path):
    ds = wids.ShardListDataset(path, cache_dir="data/")
    sampler = wids.DistributedChunkedSampler(ds, num_replicas=1, rank=0)

    wds.DataLoader(
        ds,
    )


def nsample_in_tar(tar_file: str):
    from wids.wids_mmtar import MMIndexedTar

    mmtar = MMIndexedTar(tar_file)
    print(mmtar.names(), len(mmtar.names()))
    print(len(mmtar))


if __name__ == "__main__":
    # make_loader("data/MUSLI.json")
    # nsample_in_tar('/tmp/shard.tar')
    make_loader("data/MUSLI.json")
