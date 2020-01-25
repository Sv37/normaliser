***
# normaliser.py
***

Python 3.7 module for unpacking/expanding and applying chosen  
normalisation type to the data.  
At the moment, __only z-score normalisation is available__.

The module is designed for processing very large test.tsv files  
in __chunksize=100000__ portions.


To test the module, please run in your console:

`python3 run_test.py test.tsv train.tsv z_score`

***  
  
  
## NOTE:
_Please be aware, this module is designed as a part of test  
assignment with very specific requirements, therefore should  
not be used without reconsideration._
