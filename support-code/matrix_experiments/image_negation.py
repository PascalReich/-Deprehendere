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

neg_id = np.dot(-1, np.identity(28))

fin_img = np.dot(image, neg_id)

# print(image, neg_id, np.dot(image, neg_id))

plt.imshow(image, cmap=plt.cm.binary)
plt.show()

plt.imshow(fin_img, cmap=plt.cm.binary)
plt.show()
