import numpy as np
import multiprocessing as mp
import time
import sys


matrix_1 = np.random.randint(11, size=(2, 2))
matrix_2 = np.random.randint(11, size=(2, 2))
if matrix_1.shape[1] != matrix_2.shape[0]:
    sys.exit("multiplication undefined")
num_calcs = matrix_1.shape[1] * matrix_2.shape[0]
return_dict = np.zeros((matrix_1.shape[1], matrix_2.shape[0]))


def main():
    pool = mp.Pool(processes=mp.cpu_count())

    start_normal_time = time.time()

    normal_result = np.dot(matrix_1, matrix_2)

    end_normal_time = time.time()

    normal_time = end_normal_time - start_normal_time
    print(normal_result)
    print(normal_time)
    print(matrix_1.shape[1], matrix_2.shape[0])

    global return_dict
    return_dict = np.zeros((matrix_1.shape[1], matrix_2.shape[0]))

    for i in range(matrix_1.shape[1]):
        for j in range(matrix_2.shape[0]):
            # manager = mp.Manager()
            v1 = matrix_1[i, :]
            v2 = matrix_2[:, j]
            pool.apply_async(get_dot_product, (v1, v2, [i, j]))

    pool.close()
    pool.join()
    print(return_dict)


def get_dot_product(v1, v2, target):
    return_int = 0
    for i in range(len(v2)):
        return_int += v1.item(0, i) * v2.item(i, 0)
        print(v1.item(0, i) * v2.item(i, 0))

    global return_dict
    return_dict.itemset((target[0], target[1]), return_int)


if __name__ == '__main__':
    main()
