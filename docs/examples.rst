Usage Examples
==============

Fitting Resonator Data
----------------------

.. note:: This new version of the library only supports HDF5 files from MeaVis. For the old Dataset object, use ``ooragan.old.Dataset``.

First we import data using either a :class:`Dataset <ooragan.Dataset>` for many files at once, or a :class:`File <ooragan.File>` for a single HDF5 file.

.. code::

    from ooragan import Dataset

    my_data = Dataset("path/to/folder/", attenuation_cryostat=-60)

From this object we can see the contained files simply by printing ``my_data``. We can also get the data for a single file by using the syntax ``my_data.f<i>`` where ``<i>`` can be replaced by an integer representing the file index. For example, to get the first file, ``my_data.f0``. Each File contains the data as :class:`Parameters <ooragan.Parameter>` wich have a ``range`` attribute for the raw data as well as ``name``, ``description`` and ``unit`` attributes.

.. note:: To add more parameters to extract than the ones defined in the Dataset and File classes, use the ``additional_params`` argument of those classes.

To fit the data we define a :class:`Fitter <ooragan.Fitter>` instance and call its ``fit`` method.

.. code::

    from ooragan import Fitter

    my_fitter = Fitter(my_data, savepath="path/to/saving/directory/")
    my_fitter.fit(threshold=0.5, save_fig=True)

For more fit options, see the :py:meth:`Fitter.fit <ooragan.Fitter.fit>` API documentation.

The fit results can be accessed using the same syntax as the files in a Dataset. See the :class:`FitResult <ooragan.FitResult>` API documentation for the list of attributes.

Analysing PPMS Data
-------------------

.. note::

  This section is a work in progress. Please refer to :class:`PPMSAnalysis <ooragan.PPMSAnalysis>` for help with PPMS data analysis.

Customizing Plots
-----------------

Plot generated in OORAGAN are using the `GraphingLib <https://graphinglib.org/>`_ library and every plotting method returns a `GraphingLib Figure <https://www.graphinglib.org/latest/generated/graphinglib.Figure.html#graphinglib.Figure>`_ object. This object can be used to further customize the plot by using methods implemented for Figure objects. Here is an example of a customization we can do:

.. code::

    fig = grapher.plot(
        photon=True, y_lim=(8e3, 2e5), legend_loc="outside center right", save=True
    )

    for element in fig._elements:
        element._line_width = 3
        element._errorbars_line_width = 2
        element._cap_thickness = 2
    fig.set_visual_params(
        use_latex=True,
        font_family="serif",
        font_size=18,
        axes_face_color="#E6F9FF",
        color_cycle=["#6666cc", "#df8020", "#bf4040", "#339966"],
        axes_line_width=3,
    )
    fig.set_rc_params({"xtick.major.width": 2, "ytick.major.width": 2, "xtick.major.size": 5, "ytick.major.size": 5, "xtick.minor.width": 1.5, "ytick.minor.width": 1.5, "xtick.minor.size": 4, "ytick.minor.size": 4})
    fig.set_grid(which_x="major", color="white", line_width=3)
    fig.show()

.. image:: images/custom_plot.png
  :scale: 45%

.. warning:: To obtain this exact result, matplotlib **must** have access to an installation of LaTeX on your computer.
