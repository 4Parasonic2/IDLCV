# imports the model architecture
# loads the saved weights: Use torch.load function
# loads the test set of a DatasetLoader (see train.py)
# Iterate over the test set images, generate predictions, save segmentation masks

from PIL import Image
def save_mask(array, path):
    # array should be a 2D numpy array with 0s and 1s
    # np.unique(array) == [0, 1]
    # len(np.shape(array)) == 2
    im_arr = (array*255)
    Image.fromarray(np.uint8(im_arr)).save(path)
