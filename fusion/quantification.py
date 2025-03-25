from rknn.api import RKNN
import numpy
import os
from utils import *
from tqdm import tqdm


def main():

    H = 540
    W = 960
    is_single = False
    algorithm1 = 'normal'
    method1 = 'channel'
    dtype1 = 'w8a8'
    fuse_model_path = './fusescript.pt'
    fuse_input_list = [[1, 1, H, W], [1, 1, H, W]]
    is_do_qua1 = True
    datasets1 = './datasets.txt'
    target1 = None

    vi_path = './data/vi'
    ir_path = './data/ir'
    res_path = './data/fuse'

    process_type = 'img'
    export_model_name = './fuseint8_v3.rknn'

    # fusemodel = torch.jit.load(fuse_model_path)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # fusemodel.to(device)
    # fusemodel.eval()
    fusemodel = RKNN(verbose=False)
    fusemodel.config(
        target_platform='rk3588',
        quant_img_RGB2BGR=False,
        quantized_algorithm=algorithm1,
        quantized_method=method1,
        quantized_dtype=dtype1
    )
    
    fusemodel.load_pytorch(fuse_model_path, fuse_input_list)
    if is_do_qua1:
        fusemodel.build(do_quantization=True, dataset=datasets1)
    else:
        fusemodel.build(do_quantization=False)

    fusemodel.export_rknn(export_model_name)

    

    fusemodel.init_runtime(target=target1)

    if process_type == 'img':
        img_list = os.listdir(vi_path)

        for img_name in tqdm(img_list):
            vi_img_path = os.path.join(vi_path, img_name)
            ir_img_path = os.path.join(ir_path, img_name)

            vi_img = img_read(vi_img_path, is_vis=True) # (H, W, C) float32
            ir_img = img_read(ir_img_path, is_vis=False) # (1, 1, H, W) int8

            Y, Cr, Cb = BGR2YCrCb(vi_img)# Y (1, 1, H, W) Cr (H, W) Cb (H, W) float32
            # Y = np.clip(Y*255.0, 0, 255).astype(np.uint8) # int8
            inputs = [Y, ir_img]
            fuse_img = fusemodel.inference(inputs=inputs, data_format='nchw')[0] # int8
            # Y1 = torch.from_numpy(Y).float().to(device)
            # ir_img1 = torch.from_numpy(ir_img).float().to(device)
            # with torch.no_grad():
            # fuse_img1 = fusemodel(Y1, ir_img1)
            # fuse_img = fuse_img1.cpu().numpy()
            # fuse_img = fuse_img.astype(np.float32) / 255.0 # float32
            fuse_img = YCrCb2BGR(fuse_img, Cr, Cb) # (H, W, C) float32
            save_path = os.path.join(res_path, img_name)
            img_save(fuse_img, save_path)


    # fusemodel.release()


if __name__ == '__main__':
    main()

