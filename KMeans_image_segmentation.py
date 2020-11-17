#!/usr/bin/env python
# coding: utf-8

# In[241]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
image=cv2.imread(r'C:/Users/jithi/Desktop/peppers.jpg')
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#ret, thresh=cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
#thresh_inv=cv2.bitwise_not(thresh)
plt.imshow(image)
plt.show()


# In[242]:


pixel_values=image.reshape(-1,3)
pixel_values=np.float32(pixel_values)
k=2
criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, .2)
ret, labels, centers=cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)


# In[252]:


centers=np.uint(centers)
labels=labels.flatten()


# In[253]:


segmented_data=centers[labels]


# In[254]:


segmented_image=segmented_data.reshape(image.shape)


# In[255]:


plt.imshow(segmented_image)


# In[256]:


#labels_reshape=labels.reshape(image.shape[0], image.shape[1])


# In[ ]:





# In[ ]:





# In[265]:


# disable only the cluster number 2 (turn the pixel into black)
masked_image = np.copy(image)
# convert to the shape of a vector of pixel values
masked_image = masked_image.reshape((-1, 3))
# color (i.e cluster) to disable
cluster = 0
masked_image[labels == cluster] = [0, 0, 0]
# convert back to original shape
masked_image = masked_image.reshape(image.shape)
# show the image
plt.imshow(masked_image)
plt.show()


# In[263]:


centers


# In[264]:


labels


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




