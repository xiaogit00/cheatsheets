# Deep Learning (FastAI)

### FastAI library

`from fastai.vision.all import *`


### Image Classification
**`path = untar_data(URLs.PETS)/'images'`**
- downloads data to `/Users/lei/.fastai/data/oxford-iiit-pet/images`, and returns the path object (which has many methods, but for most cases you just use print(path), which returns the string)  
- `untar_data(URL)` downloads a tar file and untars it 
- `URLs.PETS` - simply returns this URL `https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet.tgz`


### URLs datasets: