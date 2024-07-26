
![ooragan](ooragan_logo.svg)

# OORAGAN

Object Oriented Resonator Advanced Graphing and ANalysis (OORAGAN) is a library developped by me and other members of the Josephson Photonics and QIQSS research groups of the Université de Sherbrooke to fully process data from resonator measurements.

## Note

This library has been made to work with file formats **specific to the measurement setups used at the Université de Sherbrooke**. **Also**, the GitHub repository is a mirror of the development repository and no changes will be accepted on GitHub.

## Installation

This code is strctured as a library so you can install it using
```
pip install git+https://github.com/yalap13/ResonatorAnalysis.git
```

**Note** : If you are using anaconda or any other package manager, you can generally just replace the ``pip install`` with the command for installing packages for your package manager.

## Usage

Here is a simple use case :
```python
from resonatoranalysis import Dataset


path = r"C:/path/to/your/data/folder"
dataset = Dataset(path, attenuation_cryostat=-80)

# You can get a slice of this Dataset using the following
slice = dataset.slice(file_index=[1, 2], power=[-80, -100])
print(slice)
```
```bash
Found 4 files
Files :
  1. path/to/data/folder/file_1.hdf5
  2. path/to/data/folder/file_2.hdf5
  3. path/to/data/folder/file_3.hdf5
  4. path/to/data/folder/file_4.hdf5
File infos :
  File no.  Start time             Start freq. (GHz)    Stop freq. (GHz)  Power (dB)                             Mixing temp. (K)
----------  -------------------  -------------------  ------------------  -----------------------------------  ------------------
         1  2023-08-29 22:14:04              5.35653             5.36653  -100.0, -90.0, -80.0, -70.0                   0.0154368
         2  2023-08-31 01:48:44              4.89003             4.89053  -100.0, -90.0, -80.0, -70.0                   0.0136144
         3  2023-10-02 09:12:53              2                  18        -110.0, -100.0, -90.0, -80.0, -70.0
         4  2023-11-05 03:04:30              5.95844             5.96844  -110.0, -100.0, -90.0, -80.0                  0.0142297
Files :
  1. path/to/data/folder/file_1.hdf5
  2. path/to/data/folder/file_2.hdf5
File infos :
  File no.  Start time             Start freq. (GHz)    Stop freq. (GHz)  Power (dB)     Mixing temp. (K)
----------  -------------------  -------------------  ------------------  -------------  ------------------
         1  2023-08-29 22:14:04              5.35653             5.36653  -100.0, -80.0           0.0154368
         2  2023-08-31 01:48:44              4.89003             4.89053  -100.0, -80.0           0.0136144
```

ResonatorAnalysis provides a ResonatorFitter object to fit the data contained in a Dataset. Here is a usage example using the previous example as context :
```python
fitter = ra.ResonatorFitter(dataset)
# Fit only "file_1" and "file_2"
fitter.fit(file_index=[1, 2], savepic=True, write=True)
```

Running the above code will fit the data contained in the dataset for the files ``file_1`` and ``file_2``. As you can see, the ``fit`` method has the same arguments as the ``slice`` method (``file_index`` and ``power``). The path where to save the fit results and the plots can be configured with the ``savepath`` argument. By default it creates a "images" folder and a "fit_results" folder in the current working directory. Many other options exist for the ``fit`` method and all of them are detailed in the methods docstring.
