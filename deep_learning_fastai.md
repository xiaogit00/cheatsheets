# Deep Learning (FastAI)

## FastAI library

`from fastai.vision.all import *`


## Image Classification
### Downloading Images
**`path = untar_data(URLs.PETS)/'images'`** -> downloads data to */Users/lei/.fastai/data/oxford-iiit-pet/images*, and returns the path object, which is mainly that directory string
- ⁇ *untar_data(URL)* downloads a tar file and untars it 
- ⁇ *URLs.PETS* - simply returns this URL *https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet.tgz*


### URLs datasets:

1.  **ADULT_SAMPLE**: A small of the [adults
    dataset](https://archive.ics.uci.edu/ml/datasets/Adult) to predict
    whether income exceeds $50K/yr based on census data.

- **BIWI_SAMPLE**: A [BIWI kinect headpose
  database](https://www.kaggle.com/kmader/biwi-kinect-head-pose-database).
  The dataset contains over 15K images of 20 people (6 females and 14
  males - 4 people were recorded twice). For each frame, a depth image,
  the corresponding rgb image (both 640x480 pixels), and the annotation
  is provided. The head pose range covers about +-75 degrees yaw and
  +-60 degrees pitch.

1.  **CIFAR**: The famous
    [cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset
    which consists of 60000 32x32 colour images in 10 classes, with 6000
    images per class.  
2.  **COCO_SAMPLE**: A sample of the [coco
    dataset](http://cocodataset.org/#home) for object detection.
3.  **COCO_TINY**: A tiny version of the [coco
    dataset](http://cocodataset.org/#home) for object detection.

- **HUMAN_NUMBERS**: A synthetic dataset consisting of human number
  counts in text such as one, two, three, four.. Useful for
  experimenting with Language Models.

- **IMDB**: The full [IMDB sentiment analysis
  dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

- **IMDB_SAMPLE**: A sample of the full [IMDB sentiment analysis
  dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

- **ML_SAMPLE**: A movielens sample dataset for recommendation engines
  to recommend movies to users.  

- **ML_100k**: The movielens 100k dataset for recommendation engines to
  recommend movies to users.  

- **MNIST_SAMPLE**: A sample of the famous [MNIST
  dataset](http://yann.lecun.com/exdb/mnist/) consisting of handwritten
  digits.  

- **MNIST_TINY**: A tiny version of the famous [MNIST
  dataset](http://yann.lecun.com/exdb/mnist/) consisting of handwritten
  digits.  

- **MNIST_VAR_SIZE_TINY**:  

- **PLANET_SAMPLE**: A sample of the planets dataset from the Kaggle
  competition [Planet: Understanding the Amazon from
  Space](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space).

- **PLANET_TINY**: A tiny version of the planets dataset from the Kaggle
  competition [Planet: Understanding the Amazon from
  Space](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space)
  for faster experimentation and prototyping.

- **IMAGENETTE**: A smaller version of the [imagenet
  dataset](http://www.image-net.org/) pronounced just like â€˜Imagenetâ€™,
  except with a corny inauthentic French accent.

- **IMAGENETTE_160**: The 160px version of the Imagenette dataset.  

- **IMAGENETTE_320**: The 320px version of the Imagenette dataset.

- **IMAGEWOOF**: Imagewoof is a subset of 10 classes from Imagenet that
  arenâ€™t so easy to classify, since theyâ€™re all dog breeds.

- **IMAGEWOOF_160**: 160px version of the ImageWoof dataset.  

- **IMAGEWOOF_320**: 320px version of the ImageWoof dataset.

- **IMAGEWANG**: Imagewang contains Imagenette and Imagewoof combined,
  but with some twists that make it into a tricky semi-supervised
  unbalanced classification problem

- **IMAGEWANG_160**: 160px version of Imagewang.  

- **IMAGEWANG_320**: 320px version of Imagewang.

### Kaggle competition datasets

1.  **DOGS**: Image dataset consisting of dogs and cats images from
    [Dogs vs Cats kaggle
    competition](https://www.kaggle.com/c/dogs-vs-cats).

### Image Classification datasets

1.  **CALTECH_101**: Pictures of objects belonging to 101 categories.
    About 40 to 800 images per category. Most categories have about 50
    images. Collected in September 2003 by Fei-Fei Li, Marco Andreetto,
    and Marc â€™Aurelio Ranzato.
2.  **CARS**: The [Cars
    dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
    contains 16,185 images of 196 classes of cars.  
3.  **CIFAR_100**: The CIFAR-100 dataset consists of 60000 32x32 colour
    images in 100 classes, with 600 images per class.  
4.  **CUB_200_2011**: Caltech-UCSD Birds-200-2011 (CUB-200-2011) is an
    extended version of the CUB-200 dataset, with roughly double the
    number of images per class and new part location annotations
5.  **FLOWERS**: 17 category [flower
    dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/) by gathering
    images from various websites.
6.  **FOOD**:  
7.  **MNIST**: [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
    consisting of handwritten digits.  
8.  **PETS**: A 37 category [pet
    dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) with roughly
    200 images for each class.

### NLP datasets

1.  **AG_NEWS**: The AG News corpus consists of news articles from the
    AGâ€™s corpus of news articles on the web pertaining to the 4 largest
    classes. The dataset contains 30,000 training and 1,900 testing
    examples for each class.
2.  **AMAZON_REVIEWS**: This dataset contains product reviews and
    metadata from Amazon, including 142.8 million reviews spanning May
    1996 - July 2014.
3.  **AMAZON_REVIEWS_POLARITY**: Amazon reviews dataset for sentiment
    analysis.
4.  **DBPEDIA**: The DBpedia ontology dataset contains 560,000 training
    samples and 70,000 testing samples for each of 14 nonoverlapping
    classes from DBpedia.
5.  **MT_ENG_FRA**: Machine translation dataset from English to French.
6.  **SOGOU_NEWS**: [The Sogou-SRR](http://www.thuir.cn/data-srr/)
    (Search Result Relevance) dataset was constructed to support
    researches on search engine relevance estimation and ranking tasks.
7.  **WIKITEXT**: The [WikiText language modeling
    dataset](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/)
    is a collection of over 100 million tokens extracted from the set of
    verified Good and Featured articles on Wikipedia.  
8.  **WIKITEXT_TINY**: A tiny version of the WIKITEXT dataset.
9.  **YAHOO_ANSWERS**: YAHOOâ€™s question answers dataset.
10. **YELP_REVIEWS**: The [Yelp dataset](https://www.yelp.com/dataset)
    is a subset of YELP businesses, reviews, and user data for use in
    personal, educational, and academic purposes
11. **YELP_REVIEWS_POLARITY**: For sentiment classification on YELP
    reviews.

### Image localization datasets

1.  **BIWI_HEAD_POSE**: A [BIWI kinect headpose
    database](https://www.kaggle.com/kmader/biwi-kinect-head-pose-database).
    The dataset contains over 15K images of 20 people (6 females and 14
    males - 4 people were recorded twice). For each frame, a depth
    image, the corresponding rgb image (both 640x480 pixels), and the
    annotation is provided. The head pose range covers about +-75
    degrees yaw and +-60 degrees pitch.
2.  **CAMVID**: Consists of driving labelled dataset for segmentation
    type models.
3.  **CAMVID_TINY**: A tiny camvid dataset for segmentation type models.
4.  **LSUN_BEDROOMS**: [Large-scale Image
    Dataset](https://arxiv.org/abs/1506.03365) using Deep Learning with
    Humans in the Loop
5.  **PASCAL_2007**: [Pascal 2007
    dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) to
    recognize objects from a number of visual object classes in
    realistic scenes.
6.  **PASCAL_2012**: [Pascal 2012
    dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) to
    recognize objects from a number of visual object classes in
    realistic scenes.

### Audio classification

1.  **MACAQUES**: [7285 macaque coo
    calls](https://datadryad.org/stash/dataset/doi:10.5061/dryad.7f4p9)
    across 8 individuals from [Distributed acoustic cues for caller
    identity in macaque
    vocalization](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4806230).
2.  **ZEBRA_FINCH**: [3405 zebra finch
    calls](https://ndownloader.figshare.com/articles/11905533/versions/1)
    classified [across 11 call
    types](https://link.springer.com/article/10.1007/s10071-015-0933-6).
    Additional labels include name of individual making the vocalization
    and its age.

### Medical imaging datasets

1.  **SIIM_SMALL**: A smaller version of the [SIIM
    dataset](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/overview)
    where the objective is to classify pneumothorax from a set of chest
    radiographic images.

2.  **TCGA_SMALL**: A smaller version of the [TCGA-OV
    dataset](http://doi.org/10.7937/K9/TCIA.2016.NDO1MDFQ) with
    subcutaneous and visceral fat segmentations. Citations:

    Holback, C., Jarosz, R., Prior, F., Mutch, D. G., Bhosale, P.,
    Garcia, K., â€¦ Erickson, B. J. (2016). Radiology Data from The Cancer
    Genome Atlas Ovarian Cancer \[TCGA-OV\] collection. The Cancer
    Imaging Archive.
    [paper](http://doi.org/10.7937/K9/TCIA.2016.NDO1MDFQ)

    Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S,
    Phillips S, Maffitt D, Pringle M, Tarbox L, Prior F. The Cancer
    Imaging Archive (TCIA): Maintaining and Operating a Public
    Information Repository, Journal of Digital Imaging, Volume 26,
    Number 6, December, 2013, pp 1045-1057.
    [paper](https://link.springer.com/article/10.1007/s10278-013-9622-7)

### Pretrained models

1.  **OPENAI_TRANSFORMER**: The GPT2 Transformer pretrained weights.
2.  **WT103_FWD**: The WikiText-103 forward language model weights.
3.  **WT103_BWD**: The WikiText-103 backward language model weights.
