# Image Segmentation
Remove irrelevant objects from videos using FCN to solve the semantic segmentation problem. For each new image, we output pixels of the image that belong together semantically. These image segmentation algorithms relies on clustering with additional contour information and edges information.  

FCN-based semantic segmentation is used along with a bit of supervised learning.

## Setup
First have Python 3 installed and then install Python dependencies via
```
pip install -r requirements.txt
``` 

### How to run
1. Have the training data setup in a training data folder.  
2. Run `main.py` after setup.  

## What is FCN (Fully Convolutional Networks)?
Read about FCN (here)[http://deeplearning.net/tutorial/fcn_2D_segm.html]

## What is Semantic Segmentation?
Read about Semantic Segmentation (here)[https://nanonets.com/blog/how-to-do-semantic-segmentation-using-deep-learning/]

## Other useful links and github projects
https://github.com/divamgupta/image-segmentation-keras