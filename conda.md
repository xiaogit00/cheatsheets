### Conda basics
`conda list` - lists all the packages  
`conda env list` - a list of environments  

### Creating a new environment
`conda create --name ai37 python=3.11` - creates a new env called ai37 with python version 3.11.  


### Activating environments
`conda activate ai37` - activates environment ai37  
`conda deactivate` - deactivates environment 

### Installing in new environments  
`conda install numpy`  

### Errors
`PackagesNotFoundError` - Conda will typically search for packages in the standard channels, where channels are servers for ppl to host packages on. Conda-forge is community driven and a good place to start.  
Fix: `conda config --append channels conda-forge`  

### Showing channels:
`conda config --show channels`  

### Installing a package in a certain channel:
`conda install -c pytorch pytorch` - installs pytorch from the pytorch channel  