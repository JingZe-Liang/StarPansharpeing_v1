import json
import os

import ee
import geemap.core as geemap
from geemap import download_ee_image, ee_export_image

# ee.Authenticate()

ee.Initialize()
# dataset = ee.ImageCollection("EO1/HYPERION").filterDate("2016-01-01", "2017-03-01")
# img_list = dataset.toList(100)

# base_path = "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/data/EO1-HyperION"
# os.makedirs(f"{base_path}/info", exist_ok=True)
# os.makedirs(f"{base_path}/images", exist_ok=True)

# for i in range(img_list.size().getInfo()):
#     info = img_list.get(i).getInfo()

#     name = info["id"].split("/")[-1]

#     json_file = f"{base_path}/info/{name}.json"
#     with open(json_file, "w") as f:
#         json.dump(info, f, indent=2)

#     img = ee.Image(img_list.get(i))
#     valid_mask = img.select(50).gt(0)  # 选择第一个波段，找非零像素

#     valid_geometry = valid_mask.reduceToVectors(
#         geometry=img.geometry(),
#         scale=30,
#         geometryType="polygon",
#         eightConnected=False,
#         maxPixels=1e9,
#     ).geometry()
#     img_file = f"{base_path}/images/{name}.tif"

#     try:
#         # ee_export_image(img, region=img_roi, filename=img_file, scale=30)
#         download_ee_image(img, region=valid_geometry, filename=img_file, scale=30)
#         print(f"成功导出: {name}")
#     except Exception as e:
#         print(f"导出失败 {name}: {str(e)}")


dataset = ee.ImageCollection("SKYSAT/GEN-A/PUBLIC/ORTHO/RGB")
img_list = dataset.toList(100)

base_path = "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/data/SkySat"
os.makedirs(f"{base_path}/info", exist_ok=True)
os.makedirs(f"{base_path}/images", exist_ok=True)

for i in range(img_list.size().getInfo()):
    info = img_list.get(i).getInfo()

    name = info["id"].split("/")[-1]

    json_file = f"{base_path}/info/{name}.json"
    with open(json_file, "w") as f:
        json.dump(info, f, indent=2)

    # valid_geometry = valid_mask.reduceToVectors(
    #     geometry=img.geometry(),
    #     scale=30,
    #     geometryType="polygon",
    #     eightConnected=False,
    #     maxPixels=1e9,
    # ).geometry()
    img = ee.Image(img_list.get(i))
    img_roi = img.geometry().bounds()  # 获取图像的边界作为导出区域
    img_file = f"{base_path}/images/{name}.tif"

    try:
        download_ee_image(img, region=img_roi, filename=img_file, scale=2)
        # download_ee_image(img, region=valid_geometry, filename=img_file, scale=30)
        print(f"成功导出: {name}")
    except Exception as e:
        print(f"导出失败 {name}: {str(e)}")
