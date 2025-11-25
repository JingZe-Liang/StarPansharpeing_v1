import tarfile
import timeit
from collections import OrderedDict

from src.data.tar_utils import extract_tar_files_safe, read_tar_filenames_safe


def main():
    path = "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/data/BigEarthNet_S2/conditions/BigEarthNet_data_0001.tar"
    path_img = (
        "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/data/BigEarthNet_S2/hyper_images/BigEarthNet_data_0001.tar"
    )

    # no ext
    img_lst = read_tar_filenames_safe(path_img, close_tar=True)
    img_lst = dict((img.split(".")[0], None) for img in img_lst)  # ordered
    print(f"Total images in tar: {len(img_lst)}")

    # extract tar files safely
    cond_lst = read_tar_filenames_safe(path, close_tar=True)
    cond_lst = dict((cond.split(".")[0], None) for cond in cond_lst)
    print(f"Total conditions in tar: {len(cond_lst)}")

    # Find first image that doesn't have corresponding condition
    prev_k = None
    for img_key in img_lst:
        if img_key not in cond_lst:
            print(f"Image {img_key} not found in conditions tar, prev key was {prev_k}")
            break
        else:
            prev_k = img_key


if __name__ == "__main__":
    main()
