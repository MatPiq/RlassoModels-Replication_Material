conda create -n rpy2-env r-essentials r-base python=3.7
conda activate rpy2-env
conda install -c r rpy2
conda install -c conda-forge r-hdm 
pip install ../rlassopy/
pip install stata_setup
