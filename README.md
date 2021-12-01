# final-year-project
An Empirical Study on The Effectiveness of Analytical Modelling Versus Machine Learning Approaches to Software Performance Prediction

## Installation instructions:
All code for now is held in src folder
### Setup for Windows
1. Enter the src directory of your project installation using `cd <path to project>/src`
1. Install Jupyter using `py -m pip install jupyter`
1. Create virtual environment folder using `py -m venv .venv`
1. Activate your virtual environment using `cd .venv/Scripts` and then `activate.bat`. If this has worked you'll now see a (.venv) prefix in your CMD window
1. Use `cd ../..` to return to the src folder
1. Install the iPython kernel so that we can use our virtual environment in Jupyter Notebook using `py -m pip install ipykernel`
1. Add our virtual environment to the iPython kernel using `ipython kernel install --user --name=.venv`
1. Finally run your Notebook inside of our virtual environment using `py -m jupyter notebook`

