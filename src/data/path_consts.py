# Litdata path constants

RGB_PATHS = {
    "3bands_512": [
        [
            "data/AerialVG/LitData_hyper_images",
            "data/CityBench-CityData/LitData_hyper_images",
            "data/GID-GF2/LitData_hyper_images",  # slow?
            "data/LoveDA/LitData_hyper_images",
            "data/miniFrance/LitData_hyper_images",
            "data/Multispectral-Spacenet-series/LitData_hyper_images_3bands",
            "data/OpenEarthMap/LitData_hyper_images",
            "data/RefSegRS/LitData_hyper_images",
            "data/RS5M/LitData_images_train",
            "data/RS5M/LitData_images_val",
            "data/RSCaptions/LitData_hyper_images",
            "data/uavid/LitData_hyper_images",
            "data2/Disaterm3/LitData_hyper_images",
            "data/Fmow_rgb/LitData_hyper_images",
            "data2/InriaAerialLabelingDataset/LitData_hyper_images",
            "data2/RemoteSAM270k/LitData_hyper_images",
            "data2/SkyDiffusion/LitData_images",
            "data2/TEOChatlas/LitData_images_train",
            "data2/TEOChatlas/LitData_images_eval",
            "data2/UDD/LitData_hyper_images",
            "data2/VDD/LitData_hyper_images",
        ],
        dict(resize_before_transform=512, force_to_rgb=True),
    ]
}

RGB_CONDITIONS_PATHS = {
    "3bands_conditions_512": [
        [
            "data/LoveDA/LitData_conditions",
            "data/DryadHyper/LitData_captions",
            "data/GID-GF2/LitData_conditions",
            "data/Houston/LitData_conditions",
            "data/IKONOS/LitData_conditions",
            "data/MDAS-EeteS/LitData_conditions",
            "data/MDAS-HySpex/LitData_conditions",
            "data/MDAS-Optical/LitData_conditions",
            "data/MMSeg_YREB/LitData_conditions",
            "data/OpenEarthMap/LitData_conditions",
            "data/RefSegRS/LitData_conditions",
            "data/RSCaptions/LitData_conditions",
        ],
        dict(resize_before_transform=512, force_to_rgb=True),
    ]
}
MULTISPECTRAL_PATHS = {
    "4bands_512": [
        [
            "data2/Multispectral-FMow-full/LitData_hyper_images_4bands",
            "data/Gaofen1/LitData_hyper_images",
            "data/QuickBird/LitData_hyper_images",
            "data/IKONOS/LitData_hyper_images",
            "data/WorldView4/LitData_hyper_images",
            "data/MDAS-Optical/LitData_hyper_images",
            "data/EarthView/LitData_satellogic",
        ],
        dict(resize_before_transform=512),
    ],
    "8bands_512": [
        [
            "data/Multispectral-Spacenet-series/LitData_hyper_images_8bands",
            "data2/Multispectral-FMow-full/LitData_hyper_images_8bands",
            "data/WorldView2/LitData_hyper_images",
            "data/WorldView3/LitData_hyper_images",
            "data2/DCF_2019/LitData_hyper_images",
            "data/MMOT/LitData_hyper_images",
        ],
        dict(resize_before_transform=512),
    ],
    "10bands_128": [
        ["data2/TUM_128/LitData_hyper_images"],
        dict(resize_before_transform=128),
    ],
    "12bands_128": [
        ["data/MMSeg_YREB/LitData_hyper_images"],
        dict(resize_before_transform=128),
    ],
    "13bands": [
        [
            "data2/DCF_2020/LitData_hyper_images",
        ],
        dict(resize_before_transform=128),
    ],
    "bigearth_13bands_chw_128": [
        [
            "data2/BigEarthNet_S2/LitData_hyper_images",
        ],
        dict(is_hwc=False, resize_before_transform=128),
    ],
    "32bands_512": [
        [
            "data/OHS/LitData_hyper_images",
        ],
        dict(resize_before_transform=512),
    ],
    "50bands_512": [
        [
            "data/Houston/LitData_hyper_images",
        ],
        dict(resize_before_transform=512),
    ],
}


HYPERSPECTRAL_PATHS = {
    "150bands_128": [
        [
            "data/HyperGlobal/LitData_hyper_images_GF5",
        ],
        dict(resize_before_transform=128),
    ],
    "175bands_128": [
        ["data/HyperGlobal/LitData_hyper_images_EO1"],
        dict(resize_before_transform=128),
    ],
    "202bands_256": [
        [
            "data/hyspecnet11k/LitData_hyper_images",
        ],
        dict(resize_before_transform=256),
    ],
    "224bands_128": [
        [
            "data/DryadHyper/LitData_hyper_images",
        ],
        dict(resize_before_transform=128),
    ],
    "242bands_256": [
        [
            "data/MDAS-EeteS/LitData_hyper_images",
        ],
        dict(resize_before_transform=256),
    ],
    "368bands_256": [
        [
            "data/MDAS-HySpex/LitData_hyper_images",
        ],
        dict(resize_before_transform=256),
    ],
    "369bands_128": [
        [
            "data/EarthView/LitData_neon",
            # "data/EarthView/LitData_satellogic",
        ],
        dict(resize_before_transform=128),
    ],
    # "438bands_512": [
    #     [
    #         "data/MUSLI/LitData_hyper_images",
    #     ],
    #     dict(resize_before_transform=512),
    # ],
}
