from scipy import misc
import numpy as np
import matplotlib.pyplot as plt


def main():
    f = misc.face()
    plt.figure(0)
    plt.imshow(f)

    f.setflags(write=1)

    i = 0

    for row in range(len(f)):
        for column in range(len(f[row])):
            for rgb_val in range(len(f[row][column])):
                f[row][column][rgb_val] = 255 - f[row][column][rgb_val]
                i += 1

    plt.figure(1)
    plt.imshow(f)
    plt.show()
    print(i)


if __name__ == '__main__':
    main()
