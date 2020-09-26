# Information-Retrieval-2

Current project structure:
```
IR2/
 - apex
 - OpenNIR
 - src (this repo)
```
To use OpenNIR as a library: clone it in `IR2`, `` and 
```
git clone https://github.com/Georgetown-IR-Lab/OpenNIR.git
cd OpenNIR
pip install -e .
```
Because of some hard coded paths we need to run our stuff from the OpenNIR folder
```
python ../src/start.py config/conv_knrm config/antique
```

The idea is that we can put all of our code in `src` (this repo) and just extend existing OpenNIR classes, but this maybe not
possible due to its design. In the worst case we need to put our code directly into `OpenNIR/`. 

