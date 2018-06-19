import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import opening, disk, dilation, erosion, closing
import glob


def grays(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


def white_image(im):
    return cv2.bitwise_not(np.zeros(im.shape, np.uint8))


images = glob.glob('*.jpg')

for fname in images:
    image = cv2.imread(fname)
    wh = white_image(image)
    wh1 = white_image(image)
    wh2 = white_image(image)
    wh3 = white_image(image)
    wh_gray = grays(wh)
    wh_gray1 = grays(wh)
    wh_gray3 = grays(wh)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 5, sigmaColor=7, sigmaSpace=5)

    kernel_sharp = np.array((
        [-2, -2, -2],
        [-2, 17, -2],
        [-2, -2, -2]), dtype='int')
    im = cv2.filter2D(blur, -1, kernel_sharp)

    edged = cv2.Canny(im, 180, 240)
    thresh = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    # print (len(cnts))
    tnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    # print (len(tnts))
    # Thresholding is gud don't remove that for image  but no effect in case of edges so don't use for edges mapping
    cv2.drawContours(wh1, cnts, -1, 255, 1)
    cv2.drawContours(wh3, tnts, -1, 255, 1)
    # c = max(cnts, key=cv2.contourArea)
    # print (cv2.contourArea(c))
    # cv2.drawContours(wh_gray3, c, -1, 0, 1)
    o = []
    p = []
    for cnt in tnts:
        counter = 0
        cnt = np.array(cnt)
        cnt = np.reshape(cnt, (cnt.shape[0], cnt.shape[2]))
        pts = []
        # print (cv2.contourArea(cnt))
        if cv2.contourArea(cnt) > 5:
            for i in cnt:
                x, y = i
                if edged[y, x] == 255:
                    counter += 1
                    o.append(x)
                    p.append(y)
                    pts.append((x, y))
        if counter > 10:
            # print (pts)
            pts = np.array(pts)
            pts = pts.reshape(-1, 1, 2)
            # cv2.polylines(wh, [pts], True, (0, 255, 255))
            cv2.polylines(wh_gray, [pts], True, 0)
            # print (counter)
            # epsilon = 0.1 * cv2.arcLength(cnt, True)
            # approx = cv2.approxPolyDP(cnt, epsilon, True)
            # cv2.drawContours(wh_gray, approx, -1, 0, 1)

    for cnt in cnts:
        counters = 0
        cnt = np.array(cnt)
        cnt = np.reshape(cnt, (cnt.shape[0], cnt.shape[2]))
        pts = []
        print (cv2.contourArea(cnt))
        if cv2.contourArea(cnt) > 10:
            for i in cnt:
                x, y = i
                if edged[y, x] == 255:
                    counters += 1
                    o.append(x)
                    p.append(y)
                    pts.append((x, y))

        if counters > 10:
            # rect = cv2.boundingRect(cnt)
            # if rect[2] < 100 or rect[3] < 100:
            #     continue
            # x, y, w, h = rect
            # cv2.rectangle(wh_gray, (x, y), (x + w, y + h), 255, 2)
            # cv2.putText(wh, 'Moth Detected', (x + w + 10, y + h), 0, 0.3, (0, 255, 0))
            # print (pts)
            pts = np.array(pts)
            pts = pts.reshape(-1, 1, 2)
            # cv2.polylines(wh, [pts], True, (0, 255, 255))
            cv2.polylines(wh_gray1, [pts], True, 0)

    plt.figure()
    plt.title('1')
    plt.imshow(image, cmap='gray')
    # plt.figure()
    # plt.title('Grayscale Image')
    # plt.imshow(im, cmap='gray')
    # plt.figure()
    # plt.title('Canny Edge')
    # plt.imshow(edged, cmap='gray')
    # plt.figure('Thresh im')
    # plt.imshow(thresh, cmap='gray')
    # plt.figure()
    # plt.plot(o, p, 'b+')
    # plt.title('Counters edges')
    # plt.imshow(wh1, cmap='gray')
    # plt.figure()
    # plt.plot()
    # plt.imshow(wh, cmap='gray')
    # plt.figure()
    # plt.plot(o, p, 'b+')
    # plt.imshow(wh2, cmap='gray')
    # plt.figure()
    # plt.title('Counters image')
    # plt.imshow(wh3, cmap='gray')
    # plt.plot(o, p, 'k+')
    # res = erosion(wh_gray, selem=disk(6))
    # res = dilation(res, selem=disk(7))
    # mnt = cv2.findContours(wh_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    # print (mnt)
    res = closing(cv2.bitwise_not(wh_gray), selem=disk(1))
    wes = closing(cv2.bitwise_not(wh_gray1), selem=disk(1))
    # lnt = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    # wh_gray[np.where((res == 255))] = [0]
    # wh_gray[np.where((res == 0))] = [255]
    # wh_gray1[np.where((res == 255))] = [0]
    # wh_gray1[np.where((res == 0))] = [255]
    # wh_gray = erosion(wh_gray, selem=disk(2))

    # cv2.drawContours(wh2, lnt, -1, 0, 1)
    # out = ndimage.distance_transform_edt(~wh_gray)
    # out = out < 0.1 * out.max()
    # out = skeletonize(out)
    # out = morphology.binary_dilation(out, disk(3))
    # out = segmentation.clear_border(out)
    # out = out | wh_gray
    # plt.plot(o, p, 'k+')
    # plt.imshow(wh_gray, cmap='gray')
    # result = Image.fromarray(wh_gray)
    # result.save('erode.png')
    # plt.figure()
    # cv2.drawContours(wh2, lnt, -1, 0, 5)
    # plt.imshow(wh2, cmap='gray')
    # plt.figure()
    # plt.imshow(res, cmap='gray')
    # plt.figure()
    # plt.imshow(wes, cmap='gray')
    plt.figure()
    plt.imshow(wh_gray, cmap='gray')
    plt.figure()
    plt.imshow(cv2.bitwise_not(wh_gray1), cmap='gray')
    dst = cv2.bitwise_and(wh_gray, wh_gray1)
    plt.figure()
    plt.imshow(cv2.bitwise_not(dst), cmap='gray')
    plt.show()

# plt.close()
