## Ours
```json
{
  "metadata_jsonl": "src/stage2/generative/Sana/outputs_sam270k_infer/metadata.jsonl",
  "real_litdata_dir": "data2/RemoteSAM270k/LitData_hyper_images2",
  "sampler": "all",
  "cfg_ratio": "all",
  "metrics": [
    "fid",
    "is",
    "kid",
    "psnr",
    "ssim"
  ],
  "device": "cuda:0",
  "results": {
    "fid": 68.6416244506836,
    "is_mean": 6.199168682098389,
    "is_std": 0.6869083642959595,
    "kid_mean": 0.006912255194038153,
    "kid_std": 1.3276687127472542e-07,
    "psnr": 16.235318819999694,
    "ssim": 0.3397294432611088,
    "num_samples": 1000
  }
}
```


## ZImage Turbo
```json
{
  "metadata_jsonl": "src/stage2/generative/VideoX_Fun/outputs_sam270k_zimage_turbo/metadata.jsonl",
  "real_litdata_dir": "data2/RemoteSAM270k/LitData_hyper_images2",
  "condition_key": "all",
  "metrics": [
    "fid",
    "is",
    "kid",
    "psnr",
    "ssim"
  ],
  "device": "cuda:0",
  "results": {
    "fid": 107.97076416015625,
    "is_mean": 4.637654781341553,
    "is_std": 0.29649531841278076,
    "kid_mean": 0.04722781851887703,
    "kid_std": 1.2472909816096944e-07,
    "psnr": 10.796971399784088,
    "ssim": 0.21015831790596712,
    "num_samples": 1000
  }
}
```


## CRSDiff

```json
{
  "outputs_dir": "src/stage2/generative/CRS_Diff/outputs_sam270k",
  "real_litdata_dir": "data2/RemoteSAM270k/LitData_hyper_images2",
  "sampler": "ddim",
  "metrics": [
    "fid",
    "is",
    "kid",
    "psnr",
    "ssim"
  ],
  "device": "cuda:0",
  "results": {
    "fid": 103.458984375,
    "is_mean": 5.49609375,
    "is_std": 0.4445420205593109,
    "kid_mean": 0.035655248910188675,
    "kid_std": 7.77458026846034e-08,
    "psnr": 10.163461909770966,
    "ssim": 0.1518272738737578,
    "num_samples": 1000
  }
}
```
