import cv2
import numpy as np

def erode_dilate(msk, struc="ELLIPSE", size=(10, 10)):
    if struc == "RECT":      # 사각형 모양
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    elif struc == "CROSS":   # 십자가 모양
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, size)
    else:       # 원, 타원(ELLIPSE)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    msk = msk.astype(np.float32) / 255.0   # 0 or 1의 값을 가짐

    # val in 0 or 255
    # https://opencv-python.readthedocs.io/en/latest/doc/12.imageMorphological/imageMorphological.html
    dilated = cv2.dilate(msk, kernel, iterations=1) * 255
    eroded = cv2.erode(msk, kernel, iterations=1) * 255

    cnt1 = len(np.where(msk >= 0)[0])
    cnt2 = len(np.where(msk == 0)[0])
    cnt3 = len(np.where(msk == 1)[0])
    # print("all:{} bg:{} fg:{}".format(cnt1, cnt2, cnt3))
    assert(cnt1 == (cnt2 + cnt3))

    cnt1 = len(np.where(dilated >= 0)[0])
    cnt2 = len(np.where(dilated == 0)[0])
    cnt3 = len(np.where(dilated == 255)[0])
    # print("all:{} bg:{} fg:{}".format(cnt1, cnt2, cnt3))
    assert(cnt1 == (cnt2 + cnt3))

    cnt1 = len(np.where(eroded >= 0)[0])
    cnt2 = len(np.where(eroded == 0)[0])
    cnt3 = len(np.where(eroded == 255)[0])
    #  print("all:{} bg:{} fg:{}".format(cnt1, cnt2, cnt3))
    assert(cnt1 == (cnt2 + cnt3))

    res = dilated.copy()
    #res[((dilated == 255) & (msk == 0))] = 128
    res[((dilated == 255) & (eroded == 0))] = 128

    return res

# 경로는 본인의 환경(로컬)에 맞게 설정할 것!!!
def main():
    f = open('D:/Peoplespace/AI_training/project/Semantic_Human_Matting/Semantic_Human_Matting/data/list.txt')
    names = f.readlines()
    print("Images Count: {}".format(len(names)))
    for name in names:
        img_name = 'D:/Peoplespace/AI_training/project/Semantic_Human_Matting/Semantic_Human_Matting/data' + '/' + 'mattedimage' + '/' + name.strip() + ".png"
        msk_name = 'D:/Peoplespace/AI_training/project/Semantic_Human_Matting/Semantic_Human_Matting/data' + '/' + 'mask' + '/' + name.strip() + ".png"
        trimap_name = 'D:/Peoplespace/AI_training/project/Semantic_Human_Matting/Semantic_Human_Matting/data' + '/' + 'trimap' + '/' + name.strip() + ".png"
        img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)   # cv2.IMREAD_UNCHANGED(-1) -> including alpha information
        alpha = img[:,:,3]
        # print(alpha)   # value bounds between 0 and 255
        # print("Write to {}".format(msk_name))
        cv2.imwrite(msk_name, alpha)
        ret,alpha = cv2.threshold(alpha,127,255,cv2.THRESH_BINARY)  # alpha 값 중, 127보다 작은 값은 0으로, 127보다 큰 값은 최댓값으로 변경
        trimap = erode_dilate(alpha, size=(5,5))
        # cv2.imshow('alpha', alpha)
        # cv2.waitKey(0)
        # print("Write to {}".format(trimap_name))
        cv2.imwrite(trimap_name, trimap)

if __name__ == "__main__":
    main()


