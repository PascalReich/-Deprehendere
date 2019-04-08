import numpy as np
import pip
import time

try:
    import matplotlib.pyplot as plt
except:
    pip.main(['install', "matplotlib"])
    import matplotlib.pyplot as plt
import numpy as np
import gzip
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

id = np.identity(28)

fin_img = np.dot(image, id)

# print(image, neg_id, np.dot(image, neg_id))

for i in range(20):
    fin_img = np.dot(image, id)
    plt.figure(i)
    plt.imshow(image, cmap=plt.cm.binary)
    image = np.delete(image, image.shape[1] - 1, axis=1)
    id = np.identity(image.shape[1])
    # print(id)

plt.show()
