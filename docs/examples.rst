Usage Examples
==============

Fitting Resonator Data
----------------------

.. note:: This new version of the library only supports HDF5 files from MeaVis. For the old Dataset object, use ``ooragan.old.Dataset``.

First we import data using either a :class:`Dataset <ooragan.Dataset>` for many files at once, or a :class:`File <ooragan.File>` for a single HDF5 file.

.. code::

    from ooragan import Dataset

    my_data = Dataset("path/to/folder/", attenuation_cryostat=-60)

From this object we can see the contained files simply by printing ``my_data``. We can also extract the data from a single file by indexing on the dataset. For example, you can get the second file of the Dataset with ``my_data[1]`` or you can get a list of files with ``my_data[start:end]``. Each File contains the data as :class:`Parameters <ooragan.Parameter>` wich have a ``range`` attribute for the raw data as well as ``name``, ``description`` and ``unit`` attributes. To see which parameter are in a file, we can use the :py:meth:`list_paramters <ooragan.File.list_params>` method.

.. note:: To add more parameters to extract than the ones defined in the Dataset and File classes, use the ``additional_params`` argument of those classes.

To fit the data we define a :class:`Fitter <ooragan.Fitter>` instance and call its ``fit`` method.

.. code::

    from ooragan import Fitter

    my_fitter = Fitter(my_data, savepath="path/to/saving/directory/")
    my_fitter.fit(threshold=0.5, save_fig=True)

For more fit options, see the :py:meth:`Fitter.fit <ooragan.Fitter.fit>` API documentation.

The fit method creates :class:`FitResult <ooragan.FitResult>` objects for each File that's fitted. The FitResult have attributes to extract the data for quality factors, losses, resonance frequency and errors on those values. For a more complete list, check out the API documentation for the :class:`FitResult <ooragan.FitResult>`.

Plotting data
-------------

OORAGAN also defines some plotting functions to rapidly visualize the results : ::

    plot_triptych() # for the raw S21 data
    plot_quality_factors()
    plot_losses()
    plot_magnetic_field()

Those functions all return :class:`SmartFigure <graphinglib.SmartFigure>` objects from `GraphingLib <https://graphinglib.org/latest/>`_. This makes the plots `largely customizable <https://graphinglib.org/latest/handbook/smart_figure_simple.html>`_ after their creation.

Analysing PPMS Data
-------------------

.. note::

  This section is a work in progress. Please refer to :class:`PPMSAnalysis <ooragan.PPMSAnalysis>` for help with PPMS data analysis.
