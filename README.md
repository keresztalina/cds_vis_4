# Assignment 4 - Landscape recognition using pretrained CNNs
This assignment is ***Part 4*** of the portfolio exam for ***Visual Analytics S23***. The exam consists of 4 assignments in total (3 class assignments and 1 self-assigned project). This is the self-assigned project.

## 4.1. Contribution
This assignment makes use of code provided as part of the course (for example the plotting function, which has only been edited to save the produced plot). Otherwise the final code is my own. 

Here is the link to the GitHub repository containing the code for this assignment: https://github.com/keresztalina/cds_vis_4

## 4.2. Assignment description
In this self-assigned project, I wanted to see how well I could apply the methods learned during this course to a dataset that was structured differently, as well as what additional tools I could make use of. The dataset I decided to use is a collection of different landscapes found on Kaggle: https://www.kaggle.com/datasets/utkarshsaxenadn/landscape-recognition-image-dataset-12k-images. These are the possible categories, as described by the uploader:

> **Coast.** This class contains images belonging to coastal areas, or simply beaches.
> 
> **Desert.** This class contains images of desert areas such as Sahara Thar, etc.
> 
> **Forest.** This class is filled with images belonging to forest areas such as Amazon.
> 
> **Glacier.** This class consists of some amazing white images, These images belongs to glaciers. For example, the Antarctic.
> 
> **Mountains.** This class shows you the world from the top i.e. the mountain areas such as the Himalayas.

## 4.3. Methods
The purpose of this script is to use a pretrained convolutional neural network (CNN) with additional classifier layers to classify previously unseen images of landscapes into 5 possible categories (coasts, deserts, forests, glaciers and mountains) with the highest possible accuracy. 

First, the structure of the dataset is established within the script. There are 12000 images in total. The data is predivided into training, validation and test splits. Within each split, the images of each category are contained within their own folder in the following manner:

> - **Training** *(10000 images)*
>     - Coast *(2000 images)*
>     - Desert *(2000 images)*
>     - Forest *(2000 images)*
>     - Glacier *(2000 images)*
>     - Mountain *(2000 images)*
> - **Validation** *(1500 images)*
>     - Coast *(300 images)*
>     - Desert *(300 images)*
>     - Forest *(300 images)*
>     - Glacier *(300 images)*
>     - Mountain *(300 images)*
> - **Test** *(500 images)*
>     - Coast *(100 images)*
>     - Desert *(100 images)*
>     - Forest *(100 images)*
>     - Glacier *(100 images)*
>     - Mountain *(100 images)*

Second, the model is loaded. The pretrained model VGG16 is loaded without its top layers (i.e. the original classification layers), and the existing layers are set to non-trainable, as we would like to be compuationally efficient and use the existing weights. Three classifier layers are added: two hidden layers (256 and 128 neurons) with ```relu``` activation functions (in order to avoid vanishing gradients), and an output layer (15 neurons to correspond to the 15 possible labels). The learning rate of the model is also configured to an expontially decaying learning rate, where the model initially moves fast down the loss curve, then slows down in order to avoid missing the minimum.

Third, the images are preprocessed and the data is augmented. The images are rescaled and resized to 224 * 224 pixels. The data is augmented through horizontal flipping and rotation in a range of 20 degrees. The images are then prepared to be flowed from the directory in batches based on the metadata dataframes.

Then, the model is fit to the data and validated on the validation split through 10 epochs. The model's performance is tested by using to predict the labels in the test split. Finally, a classification report is made.

## 4.4 Usage
### 4.4.1. Prerequisites
This code was written and executed in the UCloud application's Coder Python interface (version 1.77.3, running Python version 3.9.2). UCloud provides virtual machines with a Linux-based operating system, therefore, the code has been optimized for Linux and may need adjustment for Windows and Mac.

### 4.4.2. Installations
1. Clone this repository somewhere on your device.
2. On the same level as your repository, create a ```/data``` folder. Download the data from https://www.kaggle.com/datasets/utkarshsaxenadn/landscape-recognition-image-dataset-12k-images into this folder. Your folder organization should look like this:

> - cds_vis_4
>     - out
>     - src
> - data
>     - test
>         - Coast
>         - Desert
>         - Forest
>         - Glacier
>         - Mountain
>     - train
>         - Coast
>         - Desert
>         - Forest
>         - Glacier
>         - Mountain
>     - val
>         - Coast
>         - Desert
>         - Forest
>         - Glacier
>         - Mountain

3. Open a terminal and navigate into the ```/cds_vis_4``` folder. Run the following lines in order to install the necessary packages:
        
        pip install --upgrade pip
        python3 -m pip install -r requirements.txt
        
### 4.4.3. Run the script
In order to run the script, make sure your current directory is still the ```/cds_vis_4``` folder. From command line, run:

        python3 src/landscape_classifier.py
        
After running the script, within the ```/out``` folder, you will find a summary of the model run, a plot of the history for diagnostics and a classification report.

## 4.5. Discussion
The classifier was set to train for 50 epochs with an early stopping function in place to prevent overtraining. In the end, it trained for 29 epochs.

When looking at the plot of the training history, it can be seen that the line for validation has become shaky and has diverged quite a bit from the line for training. This means that at one point the model began to overfit. The early stopping function was a great help with this problem, as it stopped the training when there was no further improvement and restored the optimal weights.

Overall, the landscape classifier performs significantly better than chance, reaching a mean accuracy of 77%. It is best at classifying forests, which it does with 87% accuracy, and worst at classifying mountains, which it does with 62% accuracy. Looking at the images, this can in part be explained by the fact that sometimes images contain two types of landscapes in one image, e.g. the foreground may be forest, but there may be mountains in the background, which confuses the classifier. 

Other users of this dataset have managed to achieve accuracies of 92%, however, not with VGG16. I first tried to use ResNet50, but the results were worse. Trying further classifiers and changing up some parameters may end up being able to improve accuracy. 











