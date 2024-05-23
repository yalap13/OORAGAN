# ResonatorAnalysis

This code is my adaptation of Gabriel Ouellet's code to analyse resonator measurement data also available on the JosePh Gitlab.

## Installation

This code is strctured as a library and I recommand installing it using Poetry like so
```
poetry add git+https://gitlab.gegi.usherbrooke.ca/joseph/yannick/ResonatorAnalysis.git
```
or by cloning the project
```
git clone https://gitlab.gegi.usherbrooke.ca/joseph/yannick/ResonatorAnalysis.git
```
and then installing the project with poetry
```
poetry install
```
For instructions on how to install Poetry, see [their website](https://python-poetry.org/docs/).

## Usage

Here is a simple use case :
```python
from resonatoranalysis import Dataset, fit_resonator_test
import matplotlib.pyplot as plt


path = r"C:/path/to/your/data/folder"
dataset = Dataset(path, attenuation_cryostat=-80)
files = dataset.files

for file in files:
    fit = fit_resonateur_test(
        dataset.data[file],
        list(file),
        dataset.power[file],
        savepic=True,
        write=True
    )
    plt.close()
```
It is required to manually create a folder "Images" and a folder "Fit results" to use the options `savepic=True` and `write=True`. 