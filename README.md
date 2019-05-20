A simple web app to classify fashion images.
The models were trained using Pytorch 1.0 on [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) dataset.

The model classifies the image into one of 46 categories like shorts, jeans, blouse, dress etc.

![demo](/results.gif)

### Implementation Details
Two models were pre-trained on ImageNet 
* Supervised (i.e. using labels)
* Unsupervised (without labels using Rotation as Self Supervisory signal [(paper link)](https://openreview.net/forum?id=S1v4N2l0-))

Then the models were fine-tuned on DeepFashion/train dataset.

### Credits
Following code projects were helpful for this implementation:

* [Fast AI Render sample app](https://github.com/render-examples/fastai-v3)
* [Pytorch Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam)
* [FeatureLearning RotNet](https://github.com/gidariss/FeatureLearningRotNet)

### How to replicate
* Install all the dependencies in requirements.txt + (opencv should also be installed). 
* Web app is built using Starlette framework and uvicorn as the ASGI server.
* The models were small so were included in the project. You can load your model via url also(See Fast AI Render sample code)
* To run: ```python app/server.py```

### To note:
* The models were trained on Alexnet architecture. (_TODO : explore with deeper architectures and see if gradcam visualizaitons improve_)
* only single label classification was done although many pictures in DeepFashion have multiple fashion items in a given single image. (_TODO : do multilabel classification_)
