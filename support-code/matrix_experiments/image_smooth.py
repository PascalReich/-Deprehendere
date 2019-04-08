import numpy as np
import pip
import time
from copy import deepcopy

try:
    import matplotlib.pyplot as plt
except:
    pip.main(['install', "matplotlib"])
    import matplotlib.pyplot as plt
import numpy as np
import gzip

def main():

    f = gzip.open('C:/Users/foggy/facialRecognition/train-images.gz', 'r')

    image_size = 28
    num_images = 5

    f.read(16)
    buf = f.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_size, image_size, 1)

    """for piece in range(data):
        data[piece] = np.asarray(data[piece]).squeeze()"""

    image = np.asarray(data[4]).squeeze()

    smoothed_img = deepcopy(image)

    plt.figure(0)
    plt.imshow(image, cmap=plt.cm.binary)

    # print(image, neg_id, np.dot(image, neg_id))

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            item = image.item(i, j)
            rows_to_keep = [i - 2, i - 1, i]
            columns_to_keep = [j - 2, j - 1, j]
            fin_item = mean(image[np.ix_(rows_to_keep, columns_to_keep)])
            smoothed_img.itemset((i - 1, j), fin_item)

        # print(id)
    plt.figure(1)
    plt.imshow(smoothed_img, cmap=plt.cm.binary, interpolation="bessel")

    plt.show()


def mean(matrix):
    if 15 < matrix.item(1, 1) < 240:
        values = [matrix.item(0, 1), matrix.item(1, 0), matrix.item(1, 2), matrix.item(2, 1)]
        print(matrix.item(1, 1))
        return np.mean(values)
    else:
        if matrix.item(1, 1) >= 240:
            return 255
        else:
            return 0


if __name__ == '__main__':
    main()
