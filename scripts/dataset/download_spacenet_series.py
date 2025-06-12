import os
import sys

base_dir = "/HardDisk/ZiHanCao/datasets"
assert os.path.exists(base_dir), "Base directory does not exist: {}".format(base_dir)

spacenet_cmds = dict(
    spacenet7=dict(
        train="s3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_train.tar.gz",
        val="s3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_train_csvs.tar.gz",
        test="s3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_test_public.tar.gz",
        meta="https://spacenet.ai/sn7-challenge/",
        title="SN7: Multi-Temporal Urban Development Challenge",
    ),
    spacent6=dict(
        train="s3://spacenet-dataset/spacenet/SN6_buildings/tarballs/SN6_buildings_AOI_11_Rotterdam_train.tar.gz",
        test="s3://spacenet-dataset/spacenet/SN6_buildings/tarballs/SN6_buildings_AOI_11_Rotterdam_test_public.tar.gz",
        full="s3://spacenet-dataset/AOIs/AOI_11_Rotterdam/",
        blog="https://medium.com/the-downlinq/spacenet-6-expanded-dataset-release-e1a7ddaf030",
        meta="https://spacenet.ai/sn6-challenge/",
        title="SN6: Multi-Sensor All-Weather Mapping",
    ),
    spacenet5=dict(
        train=[
            "s3://spacenet-dataset/spacenet/SN5_roads/tarballs/SN5_roads_train_AOI_7_Moscow.tar.gz",
            "s3://spacenet-dataset/spacenet/SN5_roads/tarballs/SN5_roads_train_AOI_8_Mumbai.tar.gz",
            "s3://spacenet-dataset/spacenet/SN5_roads/tarballs/SN5_roads_test_public_AOI_9_San_Juan.tar.gz",
        ],
        meta="https://spacenet.ai/sn5-challenge/",
        title="SN5: Automated Road Network Extraction and Route Travel Time Estimation from Satellite Imagery",
    ),
    spacenet4=dict(
        train="s3://spacenet-dataset/spacenet/SN4_buildings/tarballs/train/ --recursive",
        label="s3://spacenet-dataset/spacenet/SN4_buildings/tarballs/train/geojson.tar.gz",
        test="s3://spacenet-dataset/spacenet/SN4_buildings/tarballs/SN4_buildings_AOI_6_Atlanta_test_public.tar.gz",
        sample="s3://spacenet-dataset/spacenet/SN4_buildings/tarballs/summaryData.tar.gz",
        blog="https://medium.com/the-downlinq",
        meta="https://spacenet.ai/off-nadir-building-detection/",
        title="SpaceNet 4: Off-Nadir Buildings",
    ),
    spacenet3=dict(
        train=[
            "s3://spacenet-dataset/spacenet/SN3_roads/tarballs/SN3_roads_train_AOI_2_Vegas.tar.gz",
            "s3://spacenet-dataset/spacenet/SN3_roads/tarballs/SN3_roads_train_AOI_2_Vegas_geojson_roads_speed.tar.gz",
            "s3://spacenet-dataset/spacenet/SN3_roads/tarballs/SN3_roads_train_AOI_4_Shanghai.tar.gz",
            "s3://spacenet-dataset/spacenet/SN3_roads/tarballs/SN3_roads_train_AOI_4_Shanghai_geojson_roads_speed.tar.gz",
            "s3://spacenet-dataset/spacenet/SN3_roads/tarballs/SN3_roads_train_AOI_3_Paris.tar.gz",
            "s3://spacenet-dataset/spacenet/SN3_roads/tarballs/SN3_roads_train_AOI_3_Paris_geojson_roads_speed.tar.gz",
            "s3://spacenet-dataset/spacenet/SN3_roads/tarballs/SN3_roads_train_AOI_5_Khartoum.tar.gz",
            "s3://spacenet-dataset/spacenet/SN3_roads/tarballs/SN3_roads_train_AOI_5_Khartoum_geojson_roads_speed.tar.gz",
        ],
        test=[
            "s3://spacenet-dataset/spacenet/SN3_roads/tarballs/SN3_roads_test_public_AOI_2_Vegas.tar.gz",
            "s3://spacenet-dataset/spacenet/SN3_roads/tarballs/SN3_roads_test_public_AOI_4_Shanghai.tar.gz",
            "s3://spacenet-dataset/spacenet/SN3_roads/tarballs/SN3_roads_test_public_AOI_3_Paris.tar.gz",
            "s3://spacenet-dataset/spacenet/SN3_roads/tarballs/SN3_roads_test_public_AOI_5_Khartoum.tar.gz",
        ],
        sample="s3://spacenet-dataset/spacenet/SN3_roads/tarballs/SN3_roads_sample.tar.gz",
        meta="https://spacenet.ai/spacenet-roads-dataset/",
    ),
    spacenet2=dict(
        train=[
            "s3://spacenet-dataset/spacenet/SN2_buildings/tarballs/SN2_buildings_train_AOI_2_Vegas.tar.gz",
            "s3://spacenet-dataset/spacenet/SN2_buildings/tarballs/SN2_buildings_train_AOI_4_Shanghai.tar.gz",
            "s3://spacenet-dataset/spacenet/SN2_buildings/tarballs/SN2_buildings_train_AOI_3_Paris.tar.gz",
            "s3://spacenet-dataset/spacenet/SN2_buildings/tarballs/SN2_buildings_train_AOI_5_Khartoum.tar.gz",
        ],
        test=[
            "s3://spacenet-dataset/spacenet/SN2_buildings/tarballs/AOI_2_Vegas_Test_public.tar.gz",
            "s3://spacenet-dataset/spacenet/SN2_buildings/tarballs/AOI_4_Shanghai_Test_public.tar.gz",
            "s3://spacenet-dataset/spacenet/SN2_buildings/tarballs/AOI_3_Paris_Test_public.tar.gz",
            "s3://spacenet-dataset/spacenet/SN2_buildings/tarballs/AOI_5_Khartoum_Test_public.tar.gz",
        ],
        sample="s3://spacenet-dataset/spacenet/SN2_buildings/train/tarballs/ SN2_buildings_train_sample.tar.gz",
        meta="https://spacenet.ai/spacenet-buildings-dataset-v2/",
        title="SpaceNet 2: Building Detection v2",
    ),
    spacenet1=dict(
        train=[
            "s3://spacenet-dataset/spacenet/SN1_buildings/tarballs/SN1_buildings_train_AOI_1_Rio_3band.tar.gz",
            "s3://spacenet-dataset/spacenet/SN1_buildings/tarballs/SN1_buildings_train_AOI_1_Rio_8band.tar.gz",
            "s3://spacenet-dataset/spacenet/SN1_buildings/tarballs/SN1_buildings_train_AOI_1_Rio_geojson_buildings.tar.gz",
        ],
        test=[
            "s3://spacenet-dataset/spacenet/SN1_buildings/tarballs/SN1_buildings_test_AOI_1_Rio_3band.tar.gz",
            "s3://spacenet-dataset/spacenet/SN1_buildings/tarballs/SN1_buildings_test_AOI_1_Rio_8band.tar.gz",
        ],
    ),
)


def download_use_aws(urls: str | list, save_dir, no_login=True):
    if not isinstance(urls, list):
        urls = [urls]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    for url in urls:
        if no_login:
            os.system("aws s3 cp --no-sign-request {} {}".format(url, save_dir))
        else:
            os.system("aws s3 cp --no-sign-request {} {}".format(url, save_dir))


def download_spacenet_series(series_name: str, base_dir=base_dir):
    download_use_aws(
        spacenet_cmds[series_name]["train"],
        os.path.join(base_dir, series_name, "train"),
    )
    download_use_aws(
        spacenet_cmds[series_name]["test"],
        os.path.join(base_dir, series_name, "test"),
    )
    if "val" in spacenet_cmds[series_name]:
        download_use_aws(
            spacenet_cmds[series_name]["val"],
            os.path.join(base_dir, series_name, "val"),
        )
    if "label" in spacenet_cmds[series_name]:
        download_use_aws(
            spacenet_cmds[series_name]["label"],
            os.path.join(base_dir, series_name, "label"),
        )
