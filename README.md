# Person ReID

Re-Identification for person in tensorflow

## Dependencies
```
TensorFlow version >= 1.5
NumPy 1.14.0
SciPy 1.0.0
PIL 5.0.0
IPython 6.2.1
jupyter 1.0.0
```
## Model

Right now the model being used is a Siamese network, but one can change the model which is being imported in test_siamese_network_jupyter_images.ipynb to any pre-trained model. To train a model, you need to change the model function of train_helper.py file.

## Dataset

Download the MARS dataset from here:
* [MARS](https://www.kaggle.com/twoboysandhats/mars-motion-analysis-and-reidentification-set) - The dataset used for training

Next put the data into a directory with the following structure:

```
./mars/
  categories/
    0001/
      ....jpg
      ....jpg
      ....jpg
    0002/
      ....jpg
    0003/
      ....jpg
    0004/
      ....jpg
    ...
```  
You can create a TFRecord file for TensorFlow consumption. This will create a training and validation set for your dataset:
```
$ python3 create_tf_record.py --tfrecord_filename=mars --dataset_dir=/path/to/dataset/
```

## Training
The training script creates the TFRecord file and then trains the model on the generated TFRecord file.
```
$ python3 train_siamese_network.py --data /path/to/dataset/
```
The updates are stored in `./train.log/` which can be seen in TensorBoard.


## Inference
To test the model you can run the test script with two test images as such:
```
$ python3 test_siamese_network.py --img1 /path/to/image1/ --img2 /path/to/image2
```

Alternatively, you can use the two provided IPython notebooks to test the network.
The following notebook runs the test on 20 random validation set image pairs and displays the results.
```
$ jupyter notebook test_siamese_network_jupyter_validation.ipynb
```

The following notebook runs the test on the 7 test images provided in the main directory and displays the results.
```
$ jupyter notebook test_siamese_network_jupyter_images.ipynb
```

