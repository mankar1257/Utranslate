
<div align="center">
  <img src="https://github.com/mankar1257/U_translate/blob/main/images/img.png" width="650" height="375">
</div>


[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg?style=plastic)](https://badge.fury.io/py/tensorflow)
[![PyPI](https://badge.fury.io/py/tensorflow.svg)](https://badge.fury.io/py/tensorflow)



**utranslate is a Python module for machine translation built on top of TensorFlow and is distributed under the GNU General Public License v3.0.**
### Website : https://u-translate.herokuapp.com/

## Overview

There are more than 6500 languages spoken globally, and the differences in the languages make language barriers. To create a better palace for everyone, we need to remove these language barriers.


Advancement in AI and machine learning has opened thousands of possibilities for a better future. This project is an initiative to help the world lift its language barrier.

With U_translate you can build your translation system from the existing dataset or your dataset OR you can use it as a translation service for supported languages.





## Installation

### Dependencies

utanslate will requir the following pakages

* tensorflow
* tensorflow_addons
* numpy
* gensim
* requests
 
if you get the error/warning regarding **tensorflow and tensorflow_addons version compabilities** click [here](https://github.com/tensorflow/addons#python-op-compatibility-matrix) for more details 



### installation and setup
To install the current release:

```
$ pip install utranslate
```


To update utranslate to the latest version, add `--upgrade` flag to the above
commands.

to able to use the utranslate you will requried to complete the steup
Please visit the [installation Documentation](https://u-translate.herokuapp.com/install.html) for more detailed information 


## USE

### Building transletion systems 

With utranslate.build you can also build your translation system within few steps using your parallel language translation dataset between any languages.

here is the abstract example for building the traslators 
```python
#import
from utranslate.build.Custom_model import model
```

```python
#initilize and setup
U_transletor = model()
```

```python
#train
U_transletor.train()
```

```python
#save
U_transletor.save()
```

please visit the [utranslate build guids](https://u-translate.herokuapp.com/guides.html#Build) for complete instructions/tutorials for the translation system developemnt 

### Using trasletors

U_translate also provides a free neural machine translation service for the supported languages implemented in utranslate.use
useing the transletor is reletively easy following is the example for using the transletor

```python
#import
from utranslate.use.Translator import translator
```


```python
#initilize and select
U_transletor = translator()
```
~~~
avelable translation models :

  0 hi_to_en (Hindi to English)
  1 en_to_hi (English to Hindi)
  
please select the transletion model
  0
  
Starting the U_transletor this may take some time ..
  ======================================= !
    
U_transletor started 
  total time taken : 20.035918474197388 
~~~


```python
#translate
U_transletor.translate('यह सार्वभौमिक अनुवाद है')
```
~~~
this is the universal translation . 
~~~

for complete information on using the tranaletor please visit [utranslate use guids](https://u-translate.herokuapp.com/guides.html#Use)


## Contributing


#### Issues
In the case of a bug report, bugfix or suggestions, please feel free to open an issue.

#### Pull request
Pull requests are always welcome, and we will do our best to do reviews as fast as we can.


## License

This project is licensed under the [GPL License](https://github.com/mankar1257/utranslate/blob/main/LICENSE)


## Get Help
- Contact : mankarvaibhav819@gmail.com 
- If appropriate, [open an issue](https://github.com/mankar1257/utranslate/issues) on GitHub

