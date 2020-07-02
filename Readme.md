convolution
CNN layers and working
loss functions -categorical cross entrophy
optimizers -adam
architecture - our's is similar to alex net;
what is pooling
what are image histograms
what libraries used and it's use
how training done
what is supervised leanring
what is data preprocessing 


1. segmentation
        - extracting useful information from image
        - we are exctracting hand part by taking samples of color values at different places
        - and with that values we get a histogram(nothing but frequency of colors),
        - then we threshhold the image with the histogram obtained.
        - then we may get diffrent parts of images matching with that color values.
        - we calculate the areas of the each part and considers the part with max area.
2. CNN -
       - from the obtained hand portion of the image, we scale it down to 64*64 to give input to the neural net
       - expalin function of each layer with respect to our project
           -at convolution layer
                - various filters are applied to condense the features (filters are nothing but matrix of values that will be set at training).
           - at re 


yeah, coming to training 
it's its'an supervised leanrning technique
the model needs labelled input data, in this case inputs are images with respected labels.
we have used ASL dataset which contains over 3000  image for each class of labels.

for given input data and labels the models adjusts its feature maps over time in training page

in the first step we hot encode the labels that means each label is represented in binary form to represent them as output layer  such that 
if we have 5 clasess to predict 1st class represented as 10000 and second one 01000 like that.

and input image data is fed to model through 2 convolutional layers and 1 pooling layer with activation function as relu. this process takes 3 times to condense the features as much as possible.

finally the the outputs from polling layer will flattened to a vector. and connected to output layers as dense network.that means each and every neuraon in flattened layer is connected to each and every neuron in output. with 'softmax' as activation function.
which returns n predicted classes for a single input in a probabilitic manner.
