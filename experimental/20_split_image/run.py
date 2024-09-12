import random

import cv2

if __name__ == "__main__":
    im = cv2.imread("vlcsnap-2024-04-14-20h42m11s760.png")
    print(im.shape)

    y, x, _ = im.shape
    centre_x = x // 2
    centre_y = y // 2

    subsize = 336

    patches = set()

    # for each quadrant
    # work out size to fill
    gap_x = (x // 2) + (subsize // 2)
    gap_y = (y // 2) + (subsize // 2)
    count_x = (gap_x // subsize) + 1
    count_y = (gap_y // subsize) + 1
    patch_gap_x = gap_x // count_x
    patch_gap_y = gap_y // count_y

    for i in range(count_x + count_x - 1):
        for j in range(count_y + count_y - 1):
            offset_x = patch_gap_x * i
            offset_y = patch_gap_y * j
            patches.add(((offset_x, offset_y), (offset_x + subsize, offset_y + subsize)))

    for patch in sorted(patches):
        im = cv2.rectangle(im, *patch, (0, int(random.random() * 255), int(random.random() * 255)), 10)

    print(f"No. patches: {len(patches)}")

    cv2.imwrite("vlcsnap-2024-04-14-20h42m11s760.edited.png", im)
