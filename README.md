# DigitRecognition-SVC
Handwritten Digit Classifiation using support vector machines.

>Dependencies used:

OpenCV2

Scipy

ScikitLearn/SKlearn

Numpy

and some Standard libraries.



>"generateClassifier.py" generates a recognition model after training.

>"performRecognition.py" is used to classify the digits.(Takes two agrguments "-classifier" and "-i"(image)).

>"mnist-original.mat" contains all the training data.

>"deploy.bat" can be used to deploy the model.(already contains the path to the classifier,just need to add the argument for image path)
Rest are example images.


>Usage:

Use "python performRecognition.py  -classifier<PATH TO CLASSIFIER> -i<PATH TO TEST IMAGE>
  
Might throw out some weird errors at first,properly edited images with a singular color background is advised to be used.




