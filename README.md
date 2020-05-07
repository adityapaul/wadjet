# Wadjet
This is a stable implementation of the Viola-Jones object recognition algorithm, specifically designed for faces. At the moment, I've included a stable implementation of the AdaBoost, but have yet to connect it to camera.

Information on the datasets used for training can be found here: 

[UTKFace](https://susanqq.github.io/UTKFace/)

[Barcelona Dataset](http://www.cs.unc.edu/~jtighe/Papers/ECCV10/)

[Stanford Scene Segmentation Dataset](http://dags.stanford.edu/projects/scenedataset.html)

Although the datasets I used can be found on each of these links, I will try to upload the .zip files containing the image datasets I constructed.

## Flags and settings

`-t` or `--train-size`: number of samples to train on per epoch
`-e` or `--epochs`: number of epochs
`-s` or `--test-size`: number of samples to test on per epoch
`-l` or `--learners`: number of Haar learners to include in the final model
`-w` or `--window-size`: size of the image window to train on in pixels

Feedback and comments are much appreciated! Thanks!
