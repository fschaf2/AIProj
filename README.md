MNIST AI Model Comparison Project
--------------------------------
This project contains several models that attempt to classify MNIST handwritten number images by their digits. Each model has its own script that can be run on its own to test the model, or imported by another script.

Each model has a run_default() function that runs it with default settings and returns the accuracy.
Note that while most models run this function and print its results when ran directly, knn does not because it instead tests multiple k.
I have 2 base scripts, np_base.py and torch_base.py. np_base.py has basic functions that have to do with numpy, and is used by most other scripts. Specifically, it is responsible for loading data from the zip file as well as creating the confusion matrices. torch_base.py contains the base-level logic shared by all PyTorch-based models, such as training and testing procedures. 

I also have a script called full_project_tester.py, which runs a test of all the models and displays + compares their accuracies when run directly.

K-Nearest Neighbor and Naive Bayes load their data from the MNIST zip provided on Moodle, which is also included here. Once run, it creates a file mnist.npz that NumPy can access more easily. That file is also included here.

To install dependencies, please run: 
```
pip install -r requirements.txt
```
in project folder (preferably in a virtual environment).
Please contact me if you have any issues.
