'''
    test camera 

Author: Zhengwei Li
Date  : 2018/12/28
'''
import time
import numpy as np
import cv2
import torch 
import argparse
import os 
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='human matting')
parser.add_argument('--model', default='D:/Peoplespace/AI_training/project/Semantic_Human_Matting/Semantic_Human_Matting/ckpt/human_matting/model/model_obj1.pth', help='preTrained model')
parser.add_argument('--size', type=int, default=256, help='input size')
parser.add_argument('--without_gpu', action='store_true', default=True, help='no use gpu')

args = parser.parse_args()

torch.set_grad_enabled(False)

    
##################################
#----------------
if args.without_gpu:
    args.without_gpu = True
    print("use CPU !")
    device = torch.device('cpu')
else:
    if torch.cuda.is_available():
        args.without_gpu = False
        n_gpu = torch.cuda.device_count()
        print("----------------------------------------------------------")
        print("|       use GPU !      ||   Available GPU number is {} !  |".format(n_gpu))
        print("----------------------------------------------------------")

        device = torch.device('cuda:0,1')

##################################
#---------------
def load_model(args):
    print('Loading model from {}...'.format(args.model))
    if args.without_gpu:
        myModel = torch.load(args.model, map_location=lambda storage, loc: storage)
    else:
        myModel = torch.load(args.model)

    myModel.eval()
    myModel.to(device)
    
    return myModel

def seg_process(args, image, net):

    # opencv
    origin_h, origin_w, c = image.shape
    # image_resize = cv2.resize(image, (args.size,args.size), interpolation=cv2.INTER_CUBIC)
    image_resize = cv2.resize(image, (args.size,args.size), interpolation=cv2.INTER_AREA)
    image_resize = (image_resize - (104., 112., 121.,)) / 255.0

    tensor_4D = torch.FloatTensor(1, 3, args.size, args.size)
    
    tensor_4D[0,:,:,:] = torch.FloatTensor(image_resize.transpose(2,0,1))
    inputs = tensor_4D.to(device)

    t0 = time.time()

    trimap, alpha = net(inputs)

    print((time.time() - t0))  

    if args.without_gpu:
        alpha_np = alpha[0,0,:,:].data.numpy()
    else:
        alpha_np = alpha[0,0,:,:].cpu().data.numpy()

    alpha_np = cv2.resize(alpha_np, (origin_w, origin_h), interpolation=cv2.INTER_CUBIC)

    fg = np.multiply(alpha_np[..., np.newaxis], image)

    bg = image
    bg_gray = np.multiply(1-alpha_np[..., np.newaxis], image)
    bg_gray = cv2.cvtColor(bg_gray, cv2.COLOR_BGR2GRAY)

    bg[:,:,0] = bg_gray
    bg[:,:,1] = bg_gray
    bg[:,:,2] = bg_gray

    fg[fg<=0] = 0
    fg[fg>255] = 255
    fg = fg.astype(np.uint8)
    
    out = cv2.addWeighted(fg, 0.7, bg, 0.3, 0)
    out = fg + bg
    out[out<0] = 0
    out[out>255] = 255
    out = out.astype(np.uint8)

    return fg


def camera_seg(args, net):

    videoCapture = cv2.VideoCapture(0)

    while(1):
        # get a frame
        # ret, frame = videoCapture.read()
        # frame = cv2.flip(frame,1)
        frame = cv2.imread('D:/Peoplespace/AI_training/project/Semantic_Human_Matting/Semantic_Human_Matting/test/Bob1.jpg', cv2.IMREAD_COLOR)
        # frame = cv2.imread('D:/Peoplespace/AI_training/project/Semantic_Human_Matting/Semantic_Human_Matting/data/image/5.jpg', cv2.IMREAD_COLOR)
        frame_seg = seg_process(args, frame, net)
        frame_seg = cv2.resize(frame_seg, dsize=(433, 577), interpolation=cv2.INTER_AREA)

        # show a frame
        cv2.imshow("capture", frame_seg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    videoCapture.release()

def main(args):
    myModel = load_model(args)
    camera_seg(args, myModel)

if __name__ == "__main__":
    main(args)
