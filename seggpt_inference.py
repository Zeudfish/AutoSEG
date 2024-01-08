import os
import argparse
from skimage import io
import torch
import numpy as np
from copmare import calculate_image_similarity
from seggpt_engine import inference_image, inference_video
import models_seggpt


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def get_args_parser():
    parser = argparse.ArgumentParser('SegGPT inference', add_help=False)
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt',
                        default='seggpt_vit_large.pth')
    parser.add_argument('--model', type=str, help='dir to ckpt',
                        default='seggpt_vit_large_patch16_input896x448')
    parser.add_argument('--input_image', type=str, help='path to input image to be tested',
                        default=None)
    parser.add_argument('--input_video', type=str, help='path to input video to be tested',
                        default=None)
    parser.add_argument('--num_frames', type=int, help='number of prompt frames in video',
                        default=0)
    parser.add_argument('--prompt_image', type=str, nargs='+', help='path to prompt image',
                        default=None)
    parser.add_argument('--prompt_target', type=str, nargs='+', help='path to prompt target',
                        default=None)
    parser.add_argument('--seg_type', type=str, help='embedding for segmentation types', 
                        choices=['instance', 'semantic'], default='instance')
    parser.add_argument('--device', type=str, help='cuda or cpu',
                        default='cuda')
    parser.add_argument('--output_dir', type=str, help='path to output',
                        default='./cimai_data/seggpt_mask')
    parser.add_argument('--input_dir', type=str, help='path to input',
                        default='./cimai_data/image/train')
    return parser.parse_args()


def prepare_model(chkpt_dir, arch='seggpt_vit_large_patch16_input896x448', seg_type='instance'):
    # build model
    model = getattr(models_seggpt, arch)()
    model.seg_type = seg_type
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    return model

def create_label(input_floader):
    images = []
    images_folader=os.path.join(input_floader, 'image')
    mask_floader=os.path.join(input_floader, 'mask_test')


    for filename in os.listdir(images_folader):
        if filename.lower().endswith(('.bmp')):
            images_path=os.path.join(images_folader, filename)
            mask_path=os.path.join(mask_floader, os.path.splitext(filename)[0] + '.png')
            img = io.imread(images_path,as_gray=True)
            if img is not None:
                images.append((images_path, mask_path))
    return images
        


if __name__ == '__main__':
    args = get_args_parser()

    device = torch.device(args.device)
    model = prepare_model(args.ckpt_path, args.model, args.seg_type).to(device)
    print('Model loaded.')

    compare_image_array=create_label("/home/zhengm/testcode/zhufeiyu/Painter/SegGPT/SegGPT_inference/cimai_data/label/")
    
    for image_file in os.listdir(args.input_dir):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg',".bmp")):
            image_path = os.path.join(args.input_dir, image_file)
            image1=io.imread(image_path, as_gray=True)
            compare={}
            for i in range(len(compare_image_array)):
                image2_path=compare_image_array[i][0]
                image2=io.imread(image2_path, as_gray=True)
                mask_path=compare_image_array[i][1]
                mask=io.imread(mask_path, as_gray=True)
                similarity_matrix=calculate_image_similarity(image1=image1,image2=image2)
                compare[(image2_path,mask_path)]=similarity_matrix
            key_with_max_value = max(compare, key=compare.get)



            out_path = os.path.join(args.output_dir,  '.'.join(image_file.split('.')[:-1]) + '.png')


            # 调用现有的推理函数
            inference_image(model, device, image_path, [key_with_max_value[0]], [key_with_max_value[1]], out_path)

    # assert args.input_image or args.input_video and not (args.input_image and args.input_video)
    # if args.input_image is not None:
    #     assert args.prompt_image is not None and args.prompt_target is not None

    #     img_name = os.path.basename(args.input_image)
    #     out_path = os.path.join(args.output_dir, "output_" + '.'.join(img_name.split('.')[:-1]) + '.png')

    #     inference_image(model, device, args.input_image, args.prompt_image, args.prompt_target, out_path)
    
    # if args.input_video is not None:
    #     assert args.prompt_target is not None and len(args.prompt_target) == 1
    #     vid_name = os.path.basename(args.input_video)
    #     out_path = os.path.join(args.output_dir, "output_" + '.'.join(vid_name.split('.')[:-1]) + '.mp4')

    #     inference_video(model, device, args.input_video, args.num_frames, args.prompt_image, args.prompt_target, out_path)

    print('Finished.')
