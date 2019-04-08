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

plt.figure(0)
plt.imshow(image, cmap=plt.cm.binary)

# print(image, neg_id, np.dot(image, neg_id))

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        item = round(image.item(i, j)/255)*255
        image.itemset((i, j), item)


    # print(id)
plt.figure(1)
plt.imshow(image, cmap=plt.cm.binary)

plt.show()
