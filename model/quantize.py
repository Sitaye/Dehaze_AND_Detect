import os
import random
import numpy as np
from fusion.utils import img_read, BGR2YCrCb

# 图像大小
H = 480
W = 854

infrared_path = "../data/imgs/infrared"
thermal_path = "../data/imgs/thermal"
result_path = "./data/"

def main():
    os.makedirs(result_path, exist_ok=True)

    infrared_img_list = os.listdir(infrared_path)

    random.shuffle(infrared_img_list)

    img_list = infrared_img_list[:100]

    for idx, img_name in enumerate(img_list):
        infrared_img_path = os.path.join(infrared_path, img_name)
        thermal_img_path = os.path.join(thermal_path, img_name)

        infrared_img = img_read(infrared_img_path, is_single=False)
        thermal_img = img_read(thermal_img_path, is_single=True)

        Y, _, _ = BGR2YCrCb(infrared_img)

        infrared_img_save_path = os.path.join(result_path, "vi_" + str(idx))
        thermal_img_save_path = os.path.join(result_path, "ir_" + str(idx))

        np.save(infrared_img_save_path, Y)
        np.save(thermal_img_save_path, thermal_img)


if __name__ == "__main__":
    main()
