# cs231n
Solutions to Stanford CS231n Neural Network Class 2016  

---

## Environment Setup  

### Check Conda Info
conda info  

### Show all Conda environments
conda env list  

### Update Anaconda
conda update conda  
conda update anaconda  

### Create Environments
conda create -n py27 python=2 ipykernel  
conda create -n py36 python=3 ipykernel  

### Activate environment
source activate <environment>  

### Deactivate environment
source deactivate  

### Save environment to file
conda env export > puppies.yml  

### Load environment from file
conda env create -f puppies.yml  

### Open notebook: enter a conda env, then from some directory
source activate <environment>  
source activate py27  
jupyter notebook  

---
###  Assignment1

To install required packages for the CS231n assignment1 refer to the following URL  
> http://cs231n.github.io/assignments2016/assignment1/  

__Additional python setup__ 
To install additional required packages such as numpy etc:  
>
cd assignment1  
pip install -r requirements.txt  

---
###  Assignment2

__cython setup:__ 
source activate py27  
pip install Cython --install-option="--no-cython-compile"  
cd assignment2/cs231n/  
python setup.py build_ext --inplace  




