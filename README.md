
![ooragan](docs/_static/icons/ooragan_logo.svg)

# OORAGAN

Object Oriented Resonator Advanced Graphing and ANalysis (OORAGAN) is a library developped by me and other members of the Josephson Photonics and QIQSS research groups of the Université de Sherbrooke to fully process data from resonator measurements.

## Note

This library has been made to work with file formats **specific to the measurement setups used at the Université de Sherbrooke**. **Also**, the GitHub repository is a mirror of the development repository and no changes will be accepted on GitHub.

Thanks to François Cyrenne-Bergeron for the name idea.

## Table of contents

* **[Installation](#installation)**
* **[Simple example](#simple-example)**
* **[Advanced usage](#advanced-usage)**
  * [Dataset](#dataset)
    * [``Dataset.slice``](#datasetslice)
    * [``Dataset.convert_magphase_to_complex``](#datasetconvert_magphase_to_complex)
    * [``Dataset.convert_complex_to_magphase``](#datasetconvert_complex_to_magphase)
  * [ResonatorFitter](#resonatorfitter)
    * [``ResonatorFitter.fit``](#resonatorfitterfit)
  * [Grapher](#grapher)
    * [DatasetGrapher](#datasetgrapher)
      * [``DatasetGrapher.plot_mag_vs_freq``](#datasetgrapherplot_mag_vs_freq)
      * [``DatasetGrapher.plot_phase_vs_freq``](#datasetgrapherplot_phase_vs_freq)
    * [ResonatorFitterGrapher](#resonatorfittergrapher)
      * [``ResonatorFitterGrapher.plot_Qi_vs_power``](#resonatorfittergrapherplot_qi_vs_power)
      * [``ResonatorFitterGrapher.plot_Qc_vs_power``](#resonatorfittergrapherplot_qc_vs_power)
      * [``ResonatorFitterGrapher.plot_Qt_vs_power``](#resonatorfittergrapherplot_qt_vs_power)
      * [``ResonatorFitterGrapher.plot_Fshift_vs_power``](#resonatorfittergrapherplot_fshift_vs_power)
      * [``ResonatorFitterGrapher.plot_Fr_vs_power``](#resonatorfittergrapherplot_fr_vs_power)
      * [``ResonatorFitterGrapher.plot_internal_loss_vs_power``](#resonatorfittergrapherplot_internal_loss_vs_power)
    * [``load_graph_data``](#load_graph_data)

## Installation

This code is strctured as a library so you can install it using
```
pip install git+https://github.com/yalap13/OORAGAN.git
```

**Note** : If you are using anaconda or any other package manager, you can generally just replace the ``pip install`` with the command for installing packages for your package manager.

## Simple example

Here is a simple use case :
```python
import ooragan as ra


path = r"C:/path/to/your/data/folder"
dataset = ra.Dataset(path, attenuation_cryostat=-80)

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

OORAGAN provides a ResonatorFitter object to fit the data contained in a Dataset. Here is a usage example using the previous example as context :
```python
fitter = ra.ResonatorFitter(dataset)
# Fit only "file_1" and "file_2"
fitter.fit(file_index=[1, 2], savepic=True, write=True)
```

Running the above code will fit the data contained in the dataset for the files ``file_1`` and ``file_2``. As you can see, the ``fit`` method has the same arguments as the ``slice`` method (``file_index`` and ``power``). The path where to save the fit results and the plots can be configured with the ``savepath`` argument. By default it creates a "images" folder and a "fit_results" folder in the current working directory. Many other options exist for the ``fit`` method and all of them are detailed in the methods docstring.

Results can then be plotted simply :

```python
grapher = ra.grapher(fitter, save_graph_data=True)
grapher.plot_Qi_vs_power(photon=True, save=True)
```

## Advanced usage

### Dataset

The ``Dataset`` object is used to extract the data from the raw *hdf5* or *txt* files. In the background, this object creates either a ``HDF5Data`` or a ``TXTData`` object to deal with the two diffenrent data formats. Those objects are not to be used directly by the user as the ``Dataset`` creates a simpler interface.

Here is a list of the ``Dataset`` object's attribute:

<table>
  <thead>
    <tr>
      <th>Attribute</th>
      <th>Return type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>data</code></td>
      <td>dict or list</td>
      <td>Data contained in the <code>Dataset</code></td>
    </tr>
    <tr>
      <td><code>format</code></td>
      <td>str</td>
      <td>Current format of the data, either <code>"complex"</code> or <code>"magphase"</code></td>
    </tr>
    <tr>
      <td><code>cryostat_info</code></td>
      <td>dict</td>
      <td>Cryostat information. <b>Only available for hdf5 files as of now.</b></td>
    </tr>
    <tr>
      <td><code>files</code></td>
      <td>list or str</td>
      <td>Files contained in the <code>Dataset</code></td>
    </tr>
    <tr>
      <td><code>vna_average</code></td>
      <td>dict or ArrayLike</td>
      <td>VNA averaging count</td>
    </tr>
    <tr>
      <td><code>vna_bandwidth</code></td>
      <td>dict or ArrayLike</td>
      <td>VNA bandwidth</td>
    </tr>
    <tr>
      <td><code>vna_power</code></td>
      <td>dict or ArrayLike</td>
      <td>VNA output power in dBm</td>
    </tr>
    <tr>
      <td><code>variable_attenuator</code></td>
      <td>dict or ArrayLike</td>
      <td>Attenuation value of the variable attenuator. <b>Only available for hdf5 files as of now.</b></td>
    </tr>
    <tr>
      <td><code>start_time</code></td>
      <td>dict or ArrayLike</td>
      <td>Start time of the measurement</td>
    </tr>
    <tr>
      <td><code>mixing_temp</code></td>
      <td>dict or ArrayLike</td>
      <td>Temperature of the mixing stage of the fridge. <b>Only available for hdf5 files as of now.</b></td>
    </tr>
    <tr>
      <td><code>power</code></td>
      <td>dict or ArrayLike</td>
      <td>Total power includig VNA output power, attenuation on the fridge and variable attenuator.</td>
    </tr>
    <tr>
      <td><code>frequency_range</code></td>
      <td>dict</td>
      <td>Start and stop frequency of the measurement</td>
    </tr>
  </tbody>
</table>

#### ``Dataset.slice``

Creates a new ``Dataset`` containing only specified files and power values from the original ``Dataset``.

<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Type</th>
      <th>Description</th>
      <th>Required</th>
      <th>Default value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>file_index</code></td>
      <td>int or list of int</td>
      <td>Index or list of indices of files as displayed in the <code>Dataset</code>'s printed table.</td>
      <td>No</td>
      <td><code>[]</code></td>
    </tr>
    <tr>
      <td><code>power</code></td>
      <td>float or list of float</td>
      <td>Power value or list of power values as displayed in the <code>Dataset</code>'s printed table.</td>
      <td>No</td>
      <td><code>[]</code></td>
    </tr>
  </tbody>
</table>

See the [Simple example](#simple-example) section for an example of slicing a ``Dataset``.

#### ``Dataset.convert_magphase_to_complex``

Converts the data contained in the ``Dataset`` from magnitude and phase to complex format. Uses the following equation for the conversion:
$$S_{21}^\text{complex}=10^\frac{|S_{21}|}{20}e^{i\phi}$$
where the magnitude $|S_{21}|$ is in dB and the phase $\phi$ is in radians.

<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Type</th>
      <th>Description</th>
      <th>Required</th>
      <th>Default value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>deg</code></td>
      <td>bool</td>
      <td>Set to <code>True</code> if the phase is in degrees instead of radians.</td>
      <td>No</td>
      <td><code>False</code></td>
    </tr>
    <tr>
      <td><code>dBm</code></td>
      <td>bool</td>
      <td>set to <code>True</code> if the magnitude is in dBm instead of dB.</td>
      <td>No</td>
      <td><code>False</code></td>
    </tr>
  </tbody>
</table>

#### ``Dataset.convert_complex_to_magphase``

Converts the data contained in the ``Dataset`` from complex to magnitude and phase format. Uses the following to obtain the magnitude and phase from the complex signal:

$$|S_{21}|=20\cdot\log_{10}\sqrt{\mathrm{Re}(S_{21})^2+\mathrm{Im}(S_{21})^2}$$
$$\phi=\arctan\left(\frac{\mathrm{Im}(S_{21})}{\mathrm{Re}(S_{21})}\right)$$

<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Type</th>
      <th>Description</th>
      <th>Required</th>
      <th>Default value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>deg</code></td>
      <td>bool</td>
      <td>Set to <code>True</code> if the phase is in degrees instead of radians.</td>
      <td>No</td>
      <td><code>False</code></td>
    </tr>
  </tbody>
</table>

### ResonatorFitter

The ``ResonatorFitter`` object is a wrapper for a part of the [*resonator* library](https://github.com/danielflanigan/resonator) which uses the [*lmfit*](https://lmfit.github.io/lmfit-py/) fitting algorithms.

Here are the parameters of the ``ResonatorFitter`` object:

<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Type</th>
      <th>Description</th>
      <th>Required</th>
      <th>Default value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>dataset</code></td>
      <td><code>Dataset</code></td>
      <td>Dataset of the data to fit</td>
      <td>Yes</td>
      <td>N/A</td>
    </tr>
    <tr>
      <td><code>savepath</code></td>
      <td>str</td>
      <td>Path at which the "fit_results" and "fit_images" will be created to save the plots and fit results.</td>
      <td>No</td>
      <td><code>os.getcwd()</code></td>
    </tr>
  </tbody>
</table>

Here is a list of the ``ResonatorFitter`` object's attribute:

<table>
  <thead>
    <tr>
      <th>Attribute</th>
      <th>Return type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>Q_c</code></td>
      <td>dict</td>
      <td>Coupling quality factor</td>
    </tr>
    <tr>
      <td><code>Q_c_err</code></td>
      <td>dict</td>
      <td>Coupling quality factor error</td>
    </tr>
    <tr>
      <td><code>Q_i</code></td>
      <td>dict</td>
      <td>Internal quality factor</td>
    </tr>
    <tr>
      <td><code>Q_i_err</code></td>
      <td>dict</td>
      <td>Internal quality factor error</td>
    </tr>
    <tr>
      <td><code>Q_t</code></td>
      <td>dict</td>
      <td>Total quality factor</td>
    </tr>
    <tr>
      <td><code>Q_t_err</code></td>
      <td>dict</td>
      <td>Total quality factor error</td>
    </tr>
    <tr>
      <td><code>L_c</code></td>
      <td>dict</td>
      <td>Coupling loss</td>
    </tr>
    <tr>
      <td><code>L_c_err</code></td>
      <td>dict</td>
      <td>Coupling loss error</td>
    </tr>
    <tr>
      <td><code>L_i</code></td>
      <td>dict</td>
      <td>Internal loss</td>
    </tr>
    <tr>
      <td><code>L_i_err</code></td>
      <td>dict</td>
      <td>Internal loss error</td>
    </tr>
    <tr>
      <td><code>L_t</code></td>
      <td>dict</td>
      <td>Total loss</td>
    </tr>
    <tr>
      <td><code>L_t_err</code></td>
      <td>dict</td>
      <td>Total loss error</td>
    </tr>
    <tr>
      <td><code>f_r</code></td>
      <td>dict</td>
      <td>Resonance frequency</td>
    </tr>
    <tr>
      <td><code>f_r_err</code></td>
      <td>dict</td>
      <td>Resonance frequency error</td>
    </tr>
    <tr>
      <td><code>photon_number</code></td>
      <td>dict</td>
      <td>Estimated photon number</td>
    </tr>
    <tr>
      <td><code>input_power</code></td>
      <td>dict</td>
      <td>Input power applied on the transmission line</td>
    </tr>
  </tbody>
</table>

#### ``ResonatorFitter.fit``

Fits the resonance peaks with the function
$$S_{21}(f)=1-\frac{\frac{Q}{Q_c}}{1+2iQ\frac{f-f_0}{f_0}}$$
in the shunt mode and with the function
$$S_{21}(f)=-1+\frac{2\frac{Q}{Q_c}}{1+2iQ\frac{f-f_0}{f_0}}$$
in the reflection mode (as defined in the resonator library). The fit itself is performed by the lmfit library. The ``fit`` method uses a utilitary method ``_test_fit`` to verify if the maximum error tolerance on the fit results is respected. If not, the data is trimmed by a specified amount on both sides and the fit is tried again.

<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Type</th>
      <th>Description</th>
      <th>Required</th>
      <th>Default value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>file_index</code></td>
      <td>int or list of int</td>
      <td>Index or list of indices of files as displayed in the <code>Dataset</code>'s printed table.</td>
      <td>No</td>
      <td><code>[]</code></td>
    </tr>
    <tr>
      <td><code>power</code></td>
      <td>float or list of float</td>
      <td>Power values or list of power values as displayed in the <code>Dataset</code>'s printed table.</td>
      <td>No</td>
      <td><code>[]</code></td>
    </tr>
    <tr>
      <td><code>f_r</code></td>
      <td>float</td>
      <td>Resonance frequency, adds to fit parameters.</td>
      <td>No</td>
      <td><code>None</code></td>
    </tr>
    <tr>
      <td><code>couploss</code></td>
      <td>float</td>
      <td>Coupling loss, adds to the fit parameters.</td>
      <td>No</td>
      <td><code>1e-6</code></td>
    </tr>
    <tr>
      <td><code>intloss</code></td>
      <td>float</td>
      <td>Internal loss, adds to the fit parameters.</td>
      <td>No</td>
      <td><code>1e-6</code></td>
    </tr>
    <tr>
      <td><code>bg</code></td>
      <td>resonator.base.BackgroundModel</td>
      <td>Background model from the resonator library.</td>
      <td>No</td>
      <td><code>background.MagnitudePhaseDelay()</code></td>
    </tr>
    <tr>
      <td><code>savepic</code></td>
      <td>bool</td>
      <td>If <code>True</code>, saves the fit plots.</td>
      <td>No</td>
      <td><code>False</code></td>
    </tr>
    <tr>
      <td><code>showpic</code></td>
      <td>bool</td>
      <td>If <code>True</code>, the fit plots will be displayed.</td>
      <td>No</td>
      <td><code>False</code></td>
    </tr>
    <tr>
      <td><code>write</code></td>
      <td>bool</td>
      <td>If <code>True</code>, the fit results will be saved in a <em>txt</em> file.</td>
      <td>No</td>
      <td><code>False</code></td>
    </tr>
    <tr>
      <td><code>threshold</code></td>
      <td>float</td>
      <td>Error tolerance on fit results between 0 and 1.</td>
      <td>No</td>
      <td><code>0.5</code></td>
    </tr>
    <tr>
      <td><code>start</code></td>
      <td>int</td>
      <td>Index of where to start the data trimming.</td>
      <td>No</td>
      <td><code>0</code></td>
    </tr>
    <tr>
      <td><code>jump</code></td>
      <td>int</td>
      <td>Step between the data trimming.</td>
      <td>No</td>
      <td><code>10</code></td>
    </tr>
    <tr>
      <td><code>nodialog</code></td>
      <td>bool</td>
      <td>If <code>True</code>, does not display the overwriting warning popup.</td>
      <td>No</td>
      <td><code>False</code></td>
    </tr>
  </tbody>
</table>

### Grapher

The Grapher objects handle the plotting of either the raw data (``DatasetGrapher``) or the fit results (``ResonatorFitterGrapher``). In both cases, the object is created using the "factory function" ``grapher`` and providing either a ``Dataset`` or a ``ResonatorFitter``. Here are the parameters of the ``grapher`` function:

<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>type</th>
      <th>Description</th>
      <th>Required</th>
      <th>Default value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>data_object</code></td>
      <td><code>Dataset</code> or <code>ResonatorFitter</code></td>
      <td>Object containing data to plot.</td>
      <td>Yes</td>
      <td>N/A</td>
    </tr>
    <tr>
      <td><code>savepath</code></td>
      <td>str</td>
      <td>Path at which the "data_plots" or "fit_results_plots" folder to save the plots (and plot data).</td>
      <td>No</td>
      <td><code>os.getcwd()</code></td>
    </tr>
    <tr>
      <td><code>name</code></td>
      <td>str</td>
      <td>Name to add to all filenames of plots and plot data.</td>
      <td>No</td>
      <td><code>None</code></td>
    </tr>
    <tr>
      <td><code>image_type</code></td>
      <td>str</td>
      <td>Image file type for the saved plots.</td>
      <td>No</td>
      <td><code>"svg"</code></td>
    </tr>
    <tr>
      <td><code>match_pattern</code></td>
      <td>dict</td>
      <td>Dictionnary of file indices (as displayed in the <code>Dataset</code>'s printed table) associated to a label. Used to merge seperated files in a single curve on the plot. Example: <code>{"5.475 GHz": (1, 2), ...}</code></td>
      <td>No</td>
      <td><code>None</code></td>
    </tr>
    <tr>
      <td><code>save_graph_data</code></td>
      <td>bool</td>
      <td>If <code>True</code>, saves the data displayed in the plot in a <em>csv</em> file.</td>
      <td>No</td>
      <td><code>False</code></td>
    </tr>
  </tbody>
</table>

#### DatasetGrapher

##### ``DatasetGrapher.plot_mag_vs_freq``

Plots the magnitude of the signal as a function of the frequency.

<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>type</th>
      <th>Description</th>
      <th>Required</th>
      <th>Default value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>file_index</code></td>
      <td>int or list of int</td>
      <td>Index or list of indices of files as displayed in the <code>Dataset</code>'s printed table.</td>
      <td>No</td>
      <td><code>[]</code></td>
    </tr>
    <tr>
      <td><code>power</code></td>
      <td>float or list of float</td>
      <td>Power value or list of power values as displayed in the <code>Dataset</code>'s printed table.</td>
      <td>No</td>
      <td><code>[]</code></td>
    </tr>
    <tr>
      <td><code>size</code></td>
      <td>tuple</td>
      <td>Figure size</td>
      <td>No</td>
      <td><code>"default"</code> *</td>
    </tr>
    <tr>
      <td><code>title</code></td>
      <td>str</td>
      <td>Figure title</td>
      <td>No</td>
      <td><code>None</code></td>
    </tr>
    <tr>
      <td><code>show_grid</code></td>
      <td>bool</td>
      <td>If <code>True</code>, displays the grid</td>
      <td>No</td>
      <td><code>"default"</code> *</td>
    </tr>
    <tr>
      <td><code>save</code></td>
      <td>bool</td>
      <td>If <code>True</code>, saves the plot</td>
      <td>No</td>
      <td><code>True</code></td>
    </tr>
  </tbody>
</table>

\* ``"default"`` refers to the [GraphingLib default figure style](https://www.graphinglib.org/latest/handbook/figure_style_file.html#graphinglib-styles-showcase) configuration.

##### ``DatasetGrapher.plot_phase_vs_freq``

Plots the phase of the signal as a function of the frequency.

<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>type</th>
      <th>Description</th>
      <th>Required</th>
      <th>Default value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>file_index</code></td>
      <td>int or list of int</td>
      <td>Index or list of indices of files as displayed in the <code>Dataset</code>'s printed table.</td>
      <td>No</td>
      <td><code>[]</code></td>
    </tr>
    <tr>
      <td><code>power</code></td>
      <td>float or list of float</td>
      <td>Power value or list of power values as displayed in the <code>Dataset</code>'s printed table.</td>
      <td>No</td>
      <td><code>[]</code></td>
    </tr>
    <tr>
      <td><code>size</code></td>
      <td>tuple</td>
      <td>Figure size</td>
      <td>No</td>
      <td><code>"default"</code> *</td>
    </tr>
    <tr>
      <td><code>title</code></td>
      <td>str</td>
      <td>Figure title</td>
      <td>No</td>
      <td><code>None</code></td>
    </tr>
    <tr>
      <td><code>show_grid</code></td>
      <td>bool</td>
      <td>If <code>True</code>, displays the grid</td>
      <td>No</td>
      <td><code>"default"</code> *</td>
    </tr>
    <tr>
      <td><code>save</code></td>
      <td>bool</td>
      <td>If <code>True</code>, saves the plot</td>
      <td>No</td>
      <td><code>True</code></td>
    </tr>
  </tbody>
</table>

\* ``"default"`` refers to the [GraphingLib default figure style](https://www.graphinglib.org/latest/handbook/figure_style_file.html#graphinglib-styles-showcase) configuration.

#### ResonatorFitterGrapher

##### ``ResonatorFitterGrapher.plot_Qi_vs_power``

Plots the internal quality factor as a function of power.

<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>type</th>
      <th>Description</th>
      <th>Required</th>
      <th>Default value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>photon</code></td>
      <td>bool</td>
      <td>If <code>True</code>, plots as a function of photon number.</td>
      <td>No</td>
      <td><code>False</code></td>
    </tr>
    <tr>
      <td><code>x_lim</code></td>
      <td>tuple</td>
      <td>Limits for the x-axis</td>
      <td>No</td>
      <td><code>None</code></td>
    </tr>
    <tr>
      <td><code>y_lim</code></td>
      <td>tuple</td>
      <td>Limits for the y-axis</td>
      <td>No</td>
      <td><code>None</code></td>
    </tr>
    <tr>
      <td><code>size</code></td>
      <td>tuple</td>
      <td>Figure size</td>
      <td>No</td>
      <td><code>"default"</code> *</td>
    </tr>
    <tr>
      <td><code>title</code></td>
      <td>str</td>
      <td>Figure title</td>
      <td>No</td>
      <td><code>None</code></td>
    </tr>
    <tr>
      <td><code>show_grid</code></td>
      <td>bool</td>
      <td>If <code>True</code>, displays the grid</td>
      <td>No</td>
      <td><code>"default"</code> *</td>
    </tr>
    <tr>
      <td><code>legend_loc</code></td>
      <td>str</td>
      <td>Positionning of the legend. Can be one of {"best", "upper right", "upper left", "lower left", "lower right", "right", "center left", "center right", "lower center", "upper center", "center"} or {"outside upper center", "outside center right", "outside lower center"}.</td>
      <td>No</td>
      <td><code>"best"</code></td>
    </tr>
    <tr>
      <td><code>legend_cols</code></td>
      <td>int</td>
      <td>Number of columns in the legend</td>
      <td>No</td>
      <td><code>1</code></td>
    </tr>
    <tr>
      <td><code>figure_style</code></td>
      <td>str</td>
      <td>GraphingLib figure style</td>
      <td>No</td>
      <td><code>"default"</code> *</td>
    </tr>
    <tr>
      <td><code>save</code></td>
      <td>bool</td>
      <td>If <code>True</code>, saves the plot</td>
      <td>No</td>
      <td><code>True</code></td>
    </tr>
  </tbody>
</table>

\* ``"default"`` refers to the [GraphingLib default figure style](https://www.graphinglib.org/latest/handbook/figure_style_file.html#graphinglib-styles-showcase) configuration.

##### ``ResonatorFitterGrapher.plot_Qc_vs_power``

Plots the coupling quality factor as a function of power.

<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>type</th>
      <th>Description</th>
      <th>Required</th>
      <th>Default value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>photon</code></td>
      <td>bool</td>
      <td>If <code>True</code>, plots as a function of photon number.</td>
      <td>No</td>
      <td><code>False</code></td>
    </tr>
    <tr>
      <td><code>x_lim</code></td>
      <td>tuple</td>
      <td>Limits for the x-axis</td>
      <td>No</td>
      <td><code>None</code></td>
    </tr>
    <tr>
      <td><code>y_lim</code></td>
      <td>tuple</td>
      <td>Limits for the y-axis</td>
      <td>No</td>
      <td><code>None</code></td>
    </tr>
    <tr>
      <td><code>size</code></td>
      <td>tuple</td>
      <td>Figure size</td>
      <td>No</td>
      <td><code>"default"</code> *</td>
    </tr>
    <tr>
      <td><code>title</code></td>
      <td>str</td>
      <td>Figure title</td>
      <td>No</td>
      <td><code>None</code></td>
    </tr>
    <tr>
      <td><code>show_grid</code></td>
      <td>bool</td>
      <td>If <code>True</code>, displays the grid</td>
      <td>No</td>
      <td><code>"default"</code> *</td>
    </tr>
    <tr>
      <td><code>legend_loc</code></td>
      <td>str</td>
      <td>Positionning of the legend. Can be one of {"best", "upper right", "upper left", "lower left", "lower right", "right", "center left", "center right", "lower center", "upper center", "center"} or {"outside upper center", "outside center right", "outside lower center"}.</td>
      <td>No</td>
      <td><code>"best"</code></td>
    </tr>
    <tr>
      <td><code>legend_cols</code></td>
      <td>int</td>
      <td>Number of columns in the legend</td>
      <td>No</td>
      <td><code>1</code></td>
    </tr>
    <tr>
      <td><code>figure_style</code></td>
      <td>str</td>
      <td>GraphingLib figure style</td>
      <td>No</td>
      <td><code>"default"</code> *</td>
    </tr>
    <tr>
      <td><code>save</code></td>
      <td>bool</td>
      <td>If <code>True</code>, saves the plot</td>
      <td>No</td>
      <td><code>True</code></td>
    </tr>
  </tbody>
</table>

\* ``"default"`` refers to the [GraphingLib default figure style](https://www.graphinglib.org/latest/handbook/figure_style_file.html#graphinglib-styles-showcase) configuration.

##### ``ResonatorFitterGrapher.plot_Qt_vs_power``

Plots the total quality factor as a function of power.

<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>type</th>
      <th>Description</th>
      <th>Required</th>
      <th>Default value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>photon</code></td>
      <td>bool</td>
      <td>If <code>True</code>, plots as a function of photon number.</td>
      <td>No</td>
      <td><code>False</code></td>
    </tr>
    <tr>
      <td><code>x_lim</code></td>
      <td>tuple</td>
      <td>Limits for the x-axis</td>
      <td>No</td>
      <td><code>None</code></td>
    </tr>
    <tr>
      <td><code>y_lim</code></td>
      <td>tuple</td>
      <td>Limits for the y-axis</td>
      <td>No</td>
      <td><code>None</code></td>
    </tr>
    <tr>
      <td><code>size</code></td>
      <td>tuple</td>
      <td>Figure size</td>
      <td>No</td>
      <td><code>"default"</code> *</td>
    </tr>
    <tr>
      <td><code>title</code></td>
      <td>str</td>
      <td>Figure title</td>
      <td>No</td>
      <td><code>None</code></td>
    </tr>
    <tr>
      <td><code>show_grid</code></td>
      <td>bool</td>
      <td>If <code>True</code>, displays the grid</td>
      <td>No</td>
      <td><code>"default"</code> *</td>
    </tr>
    <tr>
      <td><code>legend_loc</code></td>
      <td>str</td>
      <td>Positionning of the legend. Can be one of {"best", "upper right", "upper left", "lower left", "lower right", "right", "center left", "center right", "lower center", "upper center", "center"} or {"outside upper center", "outside center right", "outside lower center"}.</td>
      <td>No</td>
      <td><code>"best"</code></td>
    </tr>
    <tr>
      <td><code>legend_cols</code></td>
      <td>int</td>
      <td>Number of columns in the legend</td>
      <td>No</td>
      <td><code>1</code></td>
    </tr>
    <tr>
      <td><code>figure_style</code></td>
      <td>str</td>
      <td>GraphingLib figure style</td>
      <td>No</td>
      <td><code>"default"</code> *</td>
    </tr>
    <tr>
      <td><code>save</code></td>
      <td>bool</td>
      <td>If <code>True</code>, saves the plot</td>
      <td>No</td>
      <td><code>True</code></td>
    </tr>
  </tbody>
</table>

\* ``"default"`` refers to the [GraphingLib default figure style](https://www.graphinglib.org/latest/handbook/figure_style_file.html#graphinglib-styles-showcase) configuration.

##### ``ResonatorFitterGrapher.plot_Fshift_vs_power``

Plots the frequency shift compared to the designed frequencies as a function of power.

<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>type</th>
      <th>Description</th>
      <th>Required</th>
      <th>Default value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>f_design</code></td>
      <td>dict</td>
      <td>Designed frequency of the attributed resonators formatted as <code>{"&lt;file_index&gt;": &lt;frequency in Hz&gt;}</code> with the <code>file_index</code> being the same index as used in the Dataset and ResonatorFitter classes.</td>
      <td>No</td>
      <td><code>False</code></td>
    </tr>
    <tr>
      <td><code>photon</code></td>
      <td>bool</td>
      <td>If <code>True</code>, plots as a function of photon number.</td>
      <td>No</td>
      <td><code>False</code></td>
    </tr>
    <tr>
      <td><code>x_lim</code></td>
      <td>tuple</td>
      <td>Limits for the x-axis</td>
      <td>No</td>
      <td><code>None</code></td>
    </tr>
    <tr>
      <td><code>y_lim</code></td>
      <td>tuple</td>
      <td>Limits for the y-axis</td>
      <td>No</td>
      <td><code>None</code></td>
    </tr>
    <tr>
      <td><code>size</code></td>
      <td>tuple</td>
      <td>Figure size</td>
      <td>No</td>
      <td><code>"default"</code> *</td>
    </tr>
    <tr>
      <td><code>title</code></td>
      <td>str</td>
      <td>Figure title</td>
      <td>No</td>
      <td><code>None</code></td>
    </tr>
    <tr>
      <td><code>show_grid</code></td>
      <td>bool</td>
      <td>If <code>True</code>, displays the grid</td>
      <td>No</td>
      <td><code>"default"</code> *</td>
    </tr>
    <tr>
      <td><code>legend_loc</code></td>
      <td>str</td>
      <td>Positionning of the legend. Can be one of {"best", "upper right", "upper left", "lower left", "lower right", "right", "center left", "center right", "lower center", "upper center", "center"} or {"outside upper center", "outside center right", "outside lower center"}.</td>
      <td>No</td>
      <td><code>"best"</code></td>
    </tr>
    <tr>
      <td><code>legend_cols</code></td>
      <td>int</td>
      <td>Number of columns in the legend</td>
      <td>No</td>
      <td><code>1</code></td>
    </tr>
    <tr>
      <td><code>figure_style</code></td>
      <td>str</td>
      <td>GraphingLib figure style</td>
      <td>No</td>
      <td><code>"default"</code> *</td>
    </tr>
    <tr>
      <td><code>save</code></td>
      <td>bool</td>
      <td>If <code>True</code>, saves the plot</td>
      <td>No</td>
      <td><code>True</code></td>
    </tr>
  </tbody>
</table>

\* ``"default"`` refers to the [GraphingLib default figure style](https://www.graphinglib.org/latest/handbook/figure_style_file.html#graphinglib-styles-showcase) configuration.

##### ``ResonatorFitterGrapher.plot_Fr_vs_power``

Plots the resonance frequency as a function of power.

<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>type</th>
      <th>Description</th>
      <th>Required</th>
      <th>Default value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>photon</code></td>
      <td>bool</td>
      <td>If <code>True</code>, plots as a function of photon number.</td>
      <td>No</td>
      <td><code>False</code></td>
    </tr>
    <tr>
      <td><code>x_lim</code></td>
      <td>tuple</td>
      <td>Limits for the x-axis</td>
      <td>No</td>
      <td><code>None</code></td>
    </tr>
    <tr>
      <td><code>y_lim</code></td>
      <td>tuple</td>
      <td>Limits for the y-axis</td>
      <td>No</td>
      <td><code>None</code></td>
    </tr>
    <tr>
      <td><code>size</code></td>
      <td>tuple</td>
      <td>Figure size</td>
      <td>No</td>
      <td><code>"default"</code> *</td>
    </tr>
    <tr>
      <td><code>title</code></td>
      <td>str</td>
      <td>Figure title</td>
      <td>No</td>
      <td><code>None</code></td>
    </tr>
    <tr>
      <td><code>show_grid</code></td>
      <td>bool</td>
      <td>If <code>True</code>, displays the grid</td>
      <td>No</td>
      <td><code>"default"</code> *</td>
    </tr>
    <tr>
      <td><code>legend_loc</code></td>
      <td>str</td>
      <td>Positionning of the legend. Can be one of {"best", "upper right", "upper left", "lower left", "lower right", "right", "center left", "center right", "lower center", "upper center", "center"} or {"outside upper center", "outside center right", "outside lower center"}.</td>
      <td>No</td>
      <td><code>"best"</code></td>
    </tr>
    <tr>
      <td><code>legend_cols</code></td>
      <td>int</td>
      <td>Number of columns in the legend</td>
      <td>No</td>
      <td><code>1</code></td>
    </tr>
    <tr>
      <td><code>figure_style</code></td>
      <td>str</td>
      <td>GraphingLib figure style</td>
      <td>No</td>
      <td><code>"default"</code> *</td>
    </tr>
    <tr>
      <td><code>save</code></td>
      <td>bool</td>
      <td>If <code>True</code>, saves the plot</td>
      <td>No</td>
      <td><code>True</code></td>
    </tr>
  </tbody>
</table>

\* ``"default"`` refers to the [GraphingLib default figure style](https://www.graphinglib.org/latest/handbook/figure_style_file.html#graphinglib-styles-showcase) configuration.

##### ``ResonatorFitterGrapher.plot_internal_loss_vs_power``

Plots the internal loss as a function of power.

<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>type</th>
      <th>Description</th>
      <th>Required</th>
      <th>Default value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>photon</code></td>
      <td>bool</td>
      <td>If <code>True</code>, plots as a function of photon number.</td>
      <td>No</td>
      <td><code>False</code></td>
    </tr>
    <tr>
      <td><code>x_lim</code></td>
      <td>tuple</td>
      <td>Limits for the x-axis</td>
      <td>No</td>
      <td><code>None</code></td>
    </tr>
    <tr>
      <td><code>y_lim</code></td>
      <td>tuple</td>
      <td>Limits for the y-axis</td>
      <td>No</td>
      <td><code>None</code></td>
    </tr>
    <tr>
      <td><code>size</code></td>
      <td>tuple</td>
      <td>Figure size</td>
      <td>No</td>
      <td><code>"default"</code> *</td>
    </tr>
    <tr>
      <td><code>title</code></td>
      <td>str</td>
      <td>Figure title</td>
      <td>No</td>
      <td><code>None</code></td>
    </tr>
    <tr>
      <td><code>show_grid</code></td>
      <td>bool</td>
      <td>If <code>True</code>, displays the grid</td>
      <td>No</td>
      <td><code>"default"</code> *</td>
    </tr>
    <tr>
      <td><code>legend_loc</code></td>
      <td>str</td>
      <td>Positionning of the legend. Can be one of {"best", "upper right", "upper left", "lower left", "lower right", "right", "center left", "center right", "lower center", "upper center", "center"} or {"outside upper center", "outside center right", "outside lower center"}.</td>
      <td>No</td>
      <td><code>"best"</code></td>
    </tr>
    <tr>
      <td><code>legend_cols</code></td>
      <td>int</td>
      <td>Number of columns in the legend</td>
      <td>No</td>
      <td><code>1</code></td>
    </tr>
    <tr>
      <td><code>figure_style</code></td>
      <td>str</td>
      <td>GraphingLib figure style</td>
      <td>No</td>
      <td><code>"default"</code> *</td>
    </tr>
    <tr>
      <td><code>save</code></td>
      <td>bool</td>
      <td>If <code>True</code>, saves the plot</td>
      <td>No</td>
      <td><code>True</code></td>
    </tr>
  </tbody>
</table>

\* ``"default"`` refers to the [GraphingLib default figure style](https://www.graphinglib.org/latest/handbook/figure_style_file.html#graphinglib-styles-showcase) configuration.

#### ``load_graph_data``

Loads the data saved in csv files by the Grapher objects and returns it as a dictionnary with label as key and NDArrays of the data.

<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>type</th>
      <th>Description</th>
      <th>Required</th>
      <th>Default value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>path</code></td>
      <td>str</td>
      <td>Complete path of the <em>csv</em> file</td>
      <td>Yes</td>
      <td>N/A</td>
    </tr>
  </tbody>
</table>
