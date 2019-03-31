import glob

import cv2 as cv
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    fourcc = cv.VideoWriter_fourcc(*'MPEG')
    out = cv.VideoWriter('output.avi', fourcc, 30.0, (669, 570))

    bg = cv.imread('images/background.jpg')
    bg = cv.resize(bg, (669, 570))
    print(bg.shape)

    for path in tqdm(glob.glob('frames/test*.png')):
        # print(path)
        # Load the image and convert it to gray scale
        img = cv.imread(path, cv.IMREAD_UNCHANGED)
        fg = img[..., 0:3]
        alpha = img[..., 3]
        alpha = alpha / 255.
        alpha = np.expand_dims(alpha, axis=-1)
        # print(alpha.shape)
        vis = alpha * fg + (1 - alpha) * bg
        vis = vis.astype(np.uint8)
        out.write(vis)

    out.release()
