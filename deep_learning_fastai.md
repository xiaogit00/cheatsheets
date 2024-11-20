# Deep Learning (FastAI)

## FastAI library

`from fastai.vision.all import *`


## Image Classification
### Downloading Images
**`path = untar_data(URLs.PETS)/'images'`** -> downloads data to */Users/lei/.fastai/data/oxford-iiit-pet/images*, and returns the path object, which is mainly that directory string
- ⁇ *untar_data(URL)* downloads a tar file and untars it 
- ⁇ *URLs.PETS* - simply returns this URL *https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet.tgz*

### Defining a label function
`def is_cat(x): return x[0].isupper()` - computer vision datasets are typically structured where the label for an image is part of the file name or most commonly the parent folder name. In this case, it's stated in the data set that "All images with 1st letter as captial are cat images; images with small first letter are dog images" - so this just checks whether first letter is caps. 

### Creating an image data loader
`dls = ImageDataLoaders.from_name_func(path, get_image_files(path), valid_pct=0.2, seed=42, label_func=is_cat, item_tfms=Resize(224))` - different data types need to be loaded differently. In this case, the ImageDataLoader takes the path, gets the image files, creates a validation set with a seed, label function, and a transformer. In this case, we apply Resize function. 
- ⁇ *item_tfms* - will apply to each item; *batch_tfms* applies to a batch of items at once using GPU, will be particularly fast. 
- ⁇ *from_name_func* - labels extracted using function applied to filename -> basically reads the file name, applies label function on file name, gets array of labels 

### Data loaders
```
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), # input and output 
    get_items=get_image_files, # what to train from: returns a list of image files
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128)
)
```
- instantiates a DataLoader that sets a couple of parameters - kind of like a pipeline object 


### Creating a convolutional neural network (CNN)
`learn = vision_learner(dls, resnet34, metrics=error_rate)` - takes in data to train on, what architecture (model) to use, and what metric to use  
- ⁇ *resnet34* - refers to number of layers of this variant, other options are 18, 50, 101, 152
- ⁇ *metric* - measures quality of model's prediction using validation set after each epoch; other options: accuracy
- ⁇ *pretrained* - a param that defaults to true, sets weights in model to values that's alrd been trained to recognize a thousand different categories across 1.3m photos, using imageNet dataset
    - we should 'nearly always use pretained model, cos it means our model is already very capable'  
    - using pre-trained, vision_learner will remove last layer as it's specifically customized to original task (imageNet dataset classification) and replace it with one or more new layers with randomized weights; last part of model known as the head 

`learn.fine_tune(1)` - 'learn' only describes the architecture of the CNN, it doesn't do anyth until we tell it to fit the data. fine_tune is a variant of fit; if you start with pretrained model, you use fine_tune. param is the number of epochs - or, how many times to look at each image. 

### Confusion Matrix, Top losses, and cleaning

`interp = ClassificationInterpretation.from_learner(learn)` - creates an interpretation object, for confusion matrix

`interp.plot_confusion_matrix()` - plots the confusion matrix 

`interp.plot_top_losses(5, nrows=1)` - plots the top losses 

`cleaner = ImageClassifierCleaner(learn)` - creates a cleaner interface, allows you to clean by top losses

`cleaner` - run cleaner on ipynb

`for idx in cleaner.delete(): cleaner.fns[idx].unlink()` - deletes 

`for idx,cat in cleaner.change(): shutil.move(str(cleaner.fns[idx]), path/cat)` - deletes 


### Uploading an image and predicting
`img = PILImage.create('bird.jpg')` - pil = python image library. simply creates an image 

`is_cat,_,probs = learn.predict(img)` returns whether is cat and probability

### Exporting & Importing
`learn.export('export.pkl)` - creates the pickle file 

`learn = load_learner('export.pkl')`

`learn.predict(im)` -> outputs a tuple ('pred, index, probability'); pred=prediction, index is the index of the category, probability is an array of prob of each category

Defining a util function to output predictions:
```
categories = ("black", "grizzly", "teddy")
def classify_image(img):
    pred, idx, probs = learn.predict(img)
    print(map(float, probs))
    return dict(zip(categories, map(float, probs)))

{'black': 1.4010183235768636e-08,
 'grizzly': 5.510236178452033e-07,
 'teddy': 0.9999994039535522}
```

### Deploying with Gradio
`import gradio as gr` 

`image = gr.Image(height=192, width=192)` - creates an image component that can be used to upload or display images

`label = gr.Label()` - displays a classification label

`examples = ['grizzly.jpg', 'teddy.jpg']` 

`intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)` - creates an interface! takes the model you've defined, define inputs and outs, throw in some examples. 

`intf.launch(inline=False)` - launches the interface at local host; to launch public link, set share=True

## Text Classification
`from fastai.text.all import *`

`dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid='test')` - defines a DataLoader, with the validation set from 'test'

`learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)` - define a text learner, using LSTM archi

`learn.fine_tune(4, 1e-2)` - 4 epochs 


## Image segmentation
`path = untar_data(URLs.CAMVID_TINY)`
```
dls = SegmentationDataLoaders.from_label_func(
    path, bs=8, fnames = get_image_files(path/"images"),
    label_func = lambda o: path/'labels'/f'{o.stem}_P{o.suffix}',
    codes = np.loadtxt(path/'codes.txt', dtype=str)
)
```

`learn = unet_learner(dls, resnet34)`

`learn.fine_tune(8)`

`learn.show_results(max_n=6, figsize=(7, 8))`
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
