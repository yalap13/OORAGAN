"""
In order to make OORAGAN lighter, the QD_Data object and necessary functions have been
directly borrowed from the pyHegel library developped by Christian Lupien at 
Université de Sherbrooke.
"""

import time
import glob
import csv
import datetime
import dateutil.tz
import io
import numpy as np
import graphinglib as gl
import os

from numpy import genfromtxt, isnan, where, array
from tabulate import tabulate
from lmfit.models import LinearModel
from loess.loess_1d import loess_1d
from scipy.constants import pi, hbar, k, e, h
from numpy.typing import ArrayLike
from typing import Optional, Literal
from graphinglib import Figure
from seaborn import color_palette


def timestamp_offset(year=None):
    """Returns the timestamp offset to add to the timestamp column
    to obtain a proper time for resistance data
    This offset is wrong when daylight saving is active.
    Because of the way it is created, it is not possible to know exactly
    when the calculation is wrong.
    """
    # MultiVu (dynacool) probably uses GetTickCount data
    # so it is not immediately affected by clock change or daylight savings.
    # but since it lasts at most 49.7 days, it must reset at some point.
    # Multivu itself does not calculate the date correctly (offset by 1) and
    # the time is also eventually wrong after daylight saving.
    # The time might be readjusted after deactivating and reactivating the resistivity option.
    #  Here dynacool multivu seems to decode the timestamp ts as
    #    datetime.timedelta(ts//86400, ts%86400)+datetime.datetime(1999,12,31)
    if year is None:
        year = time.localtime().tm_year
    offset = time.mktime(time.strptime("{}/12/31".format(year - 1), "%Y/%m/%d"))
    return offset


def timestamp_offset_log():
    """Returns the timestamp offset to add to the timestamp column
    to obtain a proper time for log data.
    This offset is wrong when daylight saving is active.
    """
    # The multivu(dynacool) timestamp calculation is wrong
    # It jumps by 3600 s when the daylight savings starts and repeats 3600 of time
    # when it turns off. It probably does time calculations assuming a day has
    # 24*3600 = 86400 s (which is not True when daylight is used).
    # The number is based on local date/time from 1899-12-30 excel, lotus epoch.
    # dynacool multivu seems to be using the following algorithms
    #  timestamp = (datetime.datetime.now()- datetime.datetime(1899,12,30)).total_seconds()
    #    note that that assumes 86400 s per day
    #  for example (datetime.datetime.now()- datetime.datetime.fromtimestamp(0)).total_seconds() can be different from
    #    time.time()
    #  going the other way:
    #    datetime.timedelta(timestamp//86400, timestamp%86400) + datetime.datetime(1899,12,30)
    unix_epoch = datetime.datetime.fromtimestamp(0)  # this returns a local time.
    t0 = datetime.datetime(1899, 12, 30)
    offset = unix_epoch - t0
    return -offset.total_seconds()


def timestamp_log_conv(timestamp):
    """Does a full conversion of all the timestamp data (can be a vector)
    to unix time.
    """
    single = False
    if not isinstance(timestamp, (list, tuple, np.ndarray)):
        single = True
        timestamp = [timestamp]
    timestamp_flat = np.asarray(timestamp).ravel()
    base = datetime.datetime(1899, 12, 30)
    day = 3600 * 24
    ret = np.empty(len(timestamp_flat))
    for i, ts in enumerate(timestamp_flat):
        if np.isnan(ts):
            ret[i] = np.nan
        else:
            dt = datetime.timedelta(ts // day, ts % day) + base
            ret[i] = time.mktime(dt.timetuple()) + dt.microsecond / 1e6
    if single:
        ret = ret[0]
    elif isinstance(timestamp, np.ndarray) and timestamp.ndim > 1:
        ret.shape = timestamp.shape
    return ret


def pick_not_nan(data):
    """For data, provide a single row of data.
    It will return the list of columns where the data is not NaN
    """
    sel = where(isnan(data) == False)[0]
    return sel


def quoted_split(string):
    """Split on , unless quoted with " """
    reader = csv.reader([string])
    return list(reader)[0]


def read_one_ppms_dat(filename, sel_i=0, nbcols=None, encoding="latin1"):
    hdrs = []
    titles = []
    i = 0
    kwargs = {}
    if nbcols is not None:
        kwargs["usecols"] = list(range(nbcols))
        kwargs["invalid_raise"] = False
    with io.open(filename, "r", encoding=encoding) as f:
        while True:
            line = f.readline().rstrip()
            i += 1
            hdrs.append(line)
            if line == "[Data]":
                break
            if i > 40:
                break
        line = f.readline().rstrip()
        i += 1
        hdrs.append(line)
        titles = quoted_split(line)
    titles = np.array(titles)
    v = genfromtxt(
        filename, skip_header=i, delimiter=",", encoding=encoding, **kwargs
    ).T
    if v.ndim == 1:
        # There was only one line:
        v = v[:, np.newaxis]
    if sel_i is None:
        sel = None
    else:
        sel = pick_not_nan(v[:, sel_i])
    return v, titles, hdrs, sel


def _glob(filename):
    if not isinstance(filename, (list, tuple, np.ndarray)):
        filename = [filename]
    filelist = []
    for fglob in filename:
        fl = glob.glob(fglob)
        fl.sort()
        filelist.extend(fl)
    if len(filelist) == 0:
        print("No file found")
        return None, False
    elif len(filelist) > 1:
        print("Found %i files" % len(filelist))
        multi = True
    else:
        multi = False
    return filelist, multi


# Instead of: v2 = genfromtxt('cooldown.dat', skip_header=31, names=None, delimiter=',').T
# could have used: v = loadtxt('cooldown.dat', skiprows=31, delimiter=',', converters={i:(lambda s: float(s.strip() or np.nan)) for i in range(23) }).T
# if the file has 23 columns
# Or if the colums to load is known: v2 = loadtxt('cooldowntransition.dat', skiprows=31, delimiter=',', usecols=sel).T
#  where sel was [1,3,4,5,6,7,8,9,10,11,14,15,16,18,19,20,21]

# To make it look completely like an ndarray,
# see:
#  https://numpy.org/doc/stable/user/basics.dispatch.html
#  https://numpy.org/doc/stable/reference/generated/numpy.lib.mixins.NDArrayOperatorsMixin.html
#  https://numpy.org/doc/stable/user/basics.subclassing.html


class QD_Data(object):
    def __init__(
        self,
        filename_or_data,
        sel_i=0,
        titles=None,
        qd_data=None,
        concat=False,
        nbcols=None,
        timestamp="auto",
        encoding="latin1",
    ):
        """provide either a numpy data array a filename or a list of filenames,
        of a Quantum Design .dat file. The filenames can have glob patterns (*,?).
        When multiple files are provided, either they are concatenated if
        concat is True (last axis) or they are combined in a 3D array
        with the middle dimension the file number, but only if they
        all have the same shape.
        When providing data, you should also provide titles
        sel_i, when not None, will select columns that are not NaN.
         It is only applied on the first file.
        The object returned can be indexed directly,
        you can also use the v attribute which refers the the data
          with selected colunmns or vr which is the raw data.
        The t attribute will be the converted timestamp.
        The trel attribute is the timestamp column minus the first value.
        timestamp is the parameter used for do_timestamp.
        titles is the selected columns,
        titles_raw is the full column names.
        headers is the full headers.
        qd_data when given, will be used for headers, titles, sel_i defaults when
          data is ndarray.
        nbcols when not None, forces to load that particular number of column and
                skip lines without enough elements.
                Use it if you receive ValueError with showing the wrong number of columns.

        Use show_titles to see the selected columns.
        Use do_sel and do_timestamp to change the column selection or the t attribute.
        """
        super(QD_Data, self).__init__()
        if isinstance(filename_or_data, np.ndarray):
            self.filenames = None
            self.vr = filename_or_data
            if qd_data is not None:
                self.filenames = qd_data.filenames
                self.headers = qd_data.headers
                self.headers_all = qd_data.headers_all
                if titles is None:
                    titles = qd_data.titles_raw
                if sel_i is None:
                    sel_i = qd_data._sel
            else:
                self.headers = None
        else:
            filenames, multi = _glob(filename_or_data)
            if filenames is None:
                return
            self.filenames = filenames
            first = True
            hdrs_all = []
            vr_all = []
            for f in filenames:
                v, _titles, hdrs, sel = read_one_ppms_dat(
                    f, sel_i=None, nbcols=nbcols, encoding=encoding
                )
                if titles is None:
                    titles = _titles
                if first:
                    self.headers = hdrs
                    v_first = v
                    first = False
                else:
                    if not concat and v.shape != v_first.shape:
                        raise RuntimeError(
                            "Files don't have the same shape. Maybe use the concat option."
                        )
                    elif concat and v.shape[:-1] != v_first.shape[:-1]:
                        raise RuntimeError(
                            "Files don't have the same number of columns shape."
                        )
                vr_all.append(v)
                hdrs_all.append(hdrs)
                if any(_titles != titles):
                    raise RuntimeError("All files do not have the same titles.")
            self.headers_all = hdrs_all
            if concat:
                v = np.concatenate(vr_all, axis=-1)
            elif not multi:
                v = vr_all[0]
            else:
                v = np.array(vr_all).swapaxes(0, 1).copy()
            self.vr = v
            self.headers = hdrs
        if titles is None:
            self.titles_raw = array(["Col_%i" % i for i in range(self.vr.shape[0])])
        else:
            self.titles_raw = array(titles)
        self._t_cache = None
        self._trel_cache = None
        self._t_conv_auto = timestamp
        self.do_sel(sel_i)

    def do_sel(self, row=0):
        """select columns for v and titles according to row content not being NaN,
        unless it is None.
        When row is a list/tuple/ndarray it will be used to select the columns.
        """
        if self.vr.ndim < 2:
            # No data, disable selection
            row = None
        if row is None:
            self._sel = row
            self.v = self.vr.copy()
            self.titles = self.titles_raw
        elif isinstance(row, slice):
            self._sel = row
            self.v = self.vr[row].copy()
            self.titles = self.titles_raw[row]
        elif isinstance(row, (list, tuple, np.ndarray)):
            self._sel = row
            self.v = self.vr[row]
            self.titles = self.titles_raw[row]
        else:
            vr = self.vr
            if len(vr.shape) == 3:
                vr = vr[:, 0]  # pick first file only.
            sel = pick_not_nan(vr[:, row])
            self._sel = sel
            self.v = self.vr[sel]
            self.titles = self.titles_raw[sel]
        self._t_cache = None
        self._trel_cache = None

    def do_timestamp(self, year="auto"):
        """generates the proper t attribute (and also returns it) from the timestamp data (column 0)
        if year is given or None, the value is used with timestamp_offset.
        if year is 'auto_year', it will try and search the header for a year,
                               if it fails it will use the current year.
        if year is 'auto' (default), it will try either timestamp_offset or timestamp_offset_log
         depending on the value. For timestamp_offset it will behave like 'auto_year'.
        if year is 'log' the timestamp_offset_log is used.
        """
        t = self[0]
        is_log = False
        if year is None:
            offset = timestamp_offset()
        elif year == "log":
            is_log = True
            offset = timestamp_offset_log()
        elif year in ["auto", "auto_year"]:
            # do not use t.min, I have seen missing time datapoints
            #   cause by an empty line in the data logs (wrapped BRlog)
            if year == "auto" and np.nanmin(t) > 10 * 365 * 24 * 3600:
                is_log = True
                offset = timestamp_offset_log()
            else:  # auto_year
                year = None
                for h in self.headers:
                    if h.startswith("FILEOPENTIME"):
                        # looks like: FILEOPENTIME,1636641706.00,11/11/2021,9:41 AM
                        # or for brlog:
                        #  FILEOPENTIME, 3846454070.154991 11/19/2021, 3:27:44 AM
                        year = int(h.split(",")[-2].split("/")[-1])
                        break
                offset = timestamp_offset(year)
        elif 1970 < year:
            offset = timestamp_offset(year)
        else:
            raise ValueError("Invalid parameter for year.")
        if is_log:
            # This is wrong, it does not handle daylight saving correctly
            # tconv = t + timestamp_offset_log()
            # But this work (however it is slower)
            tconv = timestamp_log_conv(t)
        else:
            tconv = t + offset
            # Now try to improve (it will not always be correct) for daylight savings
            lcl = time.localtime(tconv[0])
            if lcl.tm_isdst:
                tz = dateutil.tz.gettz()
                dst_offset = tz.dst(datetime.datetime(*lcl[:6])).total_seconds()
                tconv -= dst_offset
        self._t_cache = tconv
        return self._t_cache

    def show_titles(self, raw=False):
        if raw:
            t = self.titles_raw
        else:
            t = self.titles
        return list(enumerate(t))

    def __getitem__(self, indx):
        return self.v[indx]

    def __setitem__(self, indx, val):
        self.v[indx] = val

    def __iter__(self):
        return iter(self.v)

    # bring in all methods/attributes of ndarray here (except specials functions like __add__
    #  that work differently when used with + operator, those need to be added directly)
    def __getattr__(self, name):
        return getattr(self.v, name)

    @property
    def shape(self):
        return self.v.shape

    @shape.setter
    def shape(self, val):
        self.v.shape = val

    def __add__(self, val):
        """add will concatenate to data sets"""
        if not isinstance(val, QD_Data):
            raise ValueError("Can only add two Qd_Data")
        v = np.concatenate((self.vr, val.vr), axis=-1)
        nd = QD_Data(v, qd_data=self)
        nd.filenames = self.filenames + val.filenames
        nd.headers_all = [self.headers, val.headers]
        return nd

    @property
    def t(self):
        if self._t_cache is None:
            self.do_timestamp(self._t_conv_auto)
        return self._t_cache

    @property
    def trel(self):
        if self._trel_cache is None:
            self._trel_cache = self[0] - self[0, 0]
        return self._trel_cache


class PPMSAnalysis:
    """
    Loads PPMS data and implements methods to analyse the data.

    Parameters
    ----------
    path : str
        Path to the .dat file to analyse.
    start_temp : float, optional
        Start temperature of the sweep in Kelvin. Defaults to 18 K.
    end_temp : float, optional
        End temperature of the sweep in Kelvin. Defaults to 3 K.
    temp_tolerance : float, optional
        Tolerance on temperature precision when trying to find the temperature range.
        Defaults to 0 K.
    savepath : str, optional
        Path where to save the PPMSAnalysis results. Defaults to the current working
        directory.
    fname : str, optional
        File name given to all files saved. Defaults to ``None``.
    """

    def __init__(
        self,
        path: str,
        start_temp: float = 18,
        end_temp: float = 3,
        temp_tolerance: float = 0,
        savepath: str = os.getcwd(),
        fname: Optional[str] = None,
    ) -> None:
        self._data = QD_Data(filename_or_data=path)
        self._temp_dict = {}
        for title, val in zip(self._data.titles, self._data.v):
            self._temp_dict[f"{title}"] = val
        self.time_stamp = self._data.trel
        self._index = self._separator(
            temperature=self._temp_dict["Temperature (K)"],
            start=start_temp,
            end=end_temp,
            tolerance=temp_tolerance,
        )
        self._savepath = savepath
        self._fname = (
            fname
            if fname is not None
            else datetime.datetime.today().strftime("%Y-%m-%d")
        )
        self.temperature = {}
        self.resistance = {}
        self.std_dev = {}
        self.magnetic_field = {}
        for sweep, index in self._index.items():
            self.temperature[sweep] = self._temp_dict["Temperature (K)"][
                index[0] : index[1]
            ]
            self.resistance[sweep] = {
                "bridge1": self._temp_dict["Bridge 1 Resistance (Ohms)"][
                    index[0] : index[1]
                ],
                "bridge2": self._temp_dict["Bridge 2 Resistance (Ohms)"][
                    index[0] : index[1]
                ],
                "bridge3": self._temp_dict["Bridge 3 Resistance (Ohms)"][
                    index[0] : index[1]
                ],
            }
            self.std_dev[sweep] = {
                "bridge1": self._temp_dict["Bridge 1 Std. Dev. (Ohm)"][
                    index[0] : index[1]
                ],
                "bridge2": self._temp_dict["Bridge 2 Std. Dev. (Ohm)"][
                    index[0] : index[1]
                ],
                "bridge3": self._temp_dict["Bridge 3 Std. Dev. (Ohm)"][
                    index[0] : index[1]
                ],
            }
            self.magnetic_field[sweep] = self._temp_dict["Magnetic Field (Oe)"][
                index[0] : index[1]
            ]

    def _separator(
        self, temperature: ArrayLike, start: float, end: float, tolerance: float
    ) -> dict:
        """
        Finds the indices of the different temperature sweeps and stores them in a
        dictionnary.

        Parameters
        ----------
        temperature : ArrayLike
            Array of temperatures from PPMS file.
        start : int or float, optional
            Start of temperature sweep. The default is 18.
        end : int or float, optional
            End of temperature sweep. The default is 3.
        tolerance : int or float, optional
            Tolerance on temperature precision when trying to find the range. The default is 0.
        """
        index_dict = {}
        index_lst = []
        vrai = []
        for index, temp in enumerate(temperature):
            if start - tolerance <= np.round(temp, 1) <= start + tolerance:
                for index_end, temp_end in enumerate(temperature[index:]):
                    if end - tolerance <= np.round(temp_end, 1) <= end + tolerance:
                        index_lst.append([index, index_end + index])
                        break

        for i, item in enumerate(index_lst):
            if i == 0:
                vrai.append(item)
            elif i != 0:
                if item[0] - index_lst[i - 1][0] < 2:
                    continue
                else:
                    if item[1] - index_lst[i - 1][1] < 2:
                        continue
                    else:
                        vrai.append(item)

            index_dict[f"Sweep {len(index_dict) + 1}"] = vrai[-1]

        return index_dict

    def find_Tc(
        self,
        trim_index: Optional[tuple[int]] = None,
        method: str = "50",
        print_out: bool = True,
        save_to_file: bool = False,
    ) -> dict:
        """
        Finds the critical temperature using specified method.

        Parameters
        ----------
        trim_index : tuple of int, optional
            Indices at which to trim the data. Defaults to ``None``.
        method : str, optional
            Method of finding the critical temperature. Can be one of
            {"10", "50", "90", "maxgrad"}. Defaults to ``"50"``.
        print_out : bool, optional
            If ``True``, prints the results in a table. Defaults to ``True``.
        save_to_file : bool, optional
            If ``True``, saves the result in a txt file. Defaults to ``False``.
        """
        self.Tc = {}
        for sweep, temp in self.temperature.items():
            sweep_temp = {}
            for bridge, resist in self.resistance[sweep].items():
                trim_temp = (
                    temp[trim_index[0] : trim_index[1]]
                    if trim_index is not None
                    else temp
                )
                trim_resist = (
                    resist[trim_index[0] : trim_index[1]]
                    if trim_index is not None
                    else resist
                )
                r_sheet = trim_resist[0]
                if method == "10":
                    idx_tc = (np.abs(trim_resist - r_sheet * 0.1)).argmin()
                    sweep_temp[bridge] = trim_temp[idx_tc]
                elif method == "50":
                    idx_tc = (np.abs(trim_resist - r_sheet * 0.5)).argmin()
                    sweep_temp[bridge] = trim_temp[idx_tc]
                elif method == "90":
                    idx_tc = (np.abs(trim_resist - r_sheet * 0.9)).argmin()
                    sweep_temp[bridge] = trim_temp[idx_tc]
                elif method == "maxgrad":
                    x, y, err = loess_1d(trim_temp, trim_resist, degree=0, frac=0.02)
                    grad = np.gradient(y)
                    xmax = x[np.where(grad == grad.min())][0]
                    sweep_temp[bridge] = xmax
                else:
                    raise ValueError(
                        f'Method {method} is invalid, choose either "10", "50", "90" or "maxgrad"'
                    )
            self.Tc[sweep] = sweep_temp
        if print_out:
            b1 = ["Bridge 1"]
            b2 = ["Bridge 2"]
            b3 = ["Bridge 3"]
            head = ["Critical temperature (K)"]
            for sweep, vals in self.Tc.items():
                b1.append(vals["bridge1"])
                b2.append(vals["bridge2"])
                b3.append(vals["bridge3"])
                head.append(f"{int(np.mean(self.magnetic_field[sweep])/1e4)} T")
            print(tabulate([b1, b2, b3], headers=head))
        if save_to_file:
            if not os.path.exists(os.path.join(self._savepath, "ppms_analysis")):
                os.mkdir(os.path.join(self._savepath, "ppms_analysis"))
            b1 = [val["bridge1"] for _, val in self.Tc.items()]
            b2 = [val["bridge2"] for _, val in self.Tc.items()]
            b3 = [val["bridge3"] for _, val in self.Tc.items()]
            arr = np.array([b1, b2, b3])
            header = [
                f"{int(np.mean(self.magnetic_field[sweep]) / 1e4)}T"
                for sweep in self.magnetic_field.keys()
            ]
            np.savetxt(
                os.path.join(
                    self._savepath, "ppms_analysis", f"ppms_Tc_{self._fname}.txt"
                ),
                arr,
                header="\t".join(header),
                delimiter="\t",
            )
        return self.Tc

    def calculate_Lk(
        self,
        squares: Optional[float] = None,
        units: Optional[Literal["pH", "nH"]] = None,
        trim_index: Optional[tuple] = None,
        print_out: bool = True,
        save_to_file: bool = False,
    ) -> dict:
        """
        Calculates the kinetic inductance from the sheet resistance and critical
        temperature using BCS theory.

        Parameters
        ----------
        squares : float, optional
            Number of squares contained in the measured structure. If left ``None``, a
            four-point measurement on a blanket sample is assumed. Defaults to ``None``.
        units : str, optional
            Units in which to output the kinetic inductance. Either ``"pH"``, ``"nH"``
            or ``None`` for no conversion (given in H). Defaults to ``None``.
        trim_index : tuple, optional
            Indices at which to trim the data. Defaults to ``None``.
        print_out : bool, optional
            If ``True``, prints the results in a table. Defaults to ``True``.
        save_to_file : bool, optional
            If ``True``, saves the result in a txt file. Defaults to ``False``.
        """
        self.Lk = {}
        for sweep, bridges in self.Tc.items():
            Lk_temp = {}
            for bridge, tc in bridges.items():
                if squares is None:
                    r = (
                        self.resistance[sweep][bridge][0]
                        if trim_index is None
                        else self.resistance[sweep][bridge][trim_index[0]]
                    )
                    Lk_temp[bridge] = (hbar * pi / np.log(2) * r) / (
                        pi * 1.764 * k * tc
                    )
                else:
                    r = (
                        self.resistance[sweep][bridge][0]
                        if trim_index is None
                        else self.resistance[sweep][bridge][trim_index[0]]
                    )
                    Lk_temp[bridge] = (hbar * r / squares) / (1.764 * pi * k * tc)
                if units is not None:
                    if units == "pH":
                        Lk_temp[bridge] *= 1e12
                    elif units == "nH":
                        Lk_temp[bridge] *= 1e9
                    else:
                        raise ValueError('Units can be "pH", "nH" or None')
            self.Lk[sweep] = Lk_temp
        if print_out:
            b1 = ["Bridge 1"]
            b2 = ["Bridge 2"]
            b3 = ["Bridge 3"]
            head = (
                [f"Kinetic inductance ({units})"]
                if units is not None
                else ["Kinetic inductance (H)"]
            )
            for sweep, vals in self.Lk.items():
                b1.append(vals["bridge1"])
                b2.append(vals["bridge2"])
                b3.append(vals["bridge3"])
                head.append(f"{int(np.mean(self.magnetic_field[sweep])/1e4)} T")
            print(tabulate([b1, b2, b3], headers=head))
        if save_to_file:
            if not os.path.exists(os.path.join(self._savepath, "ppms_analysis")):
                os.mkdir(os.path.join(self._savepath, "ppms_analysis"))
            b1 = [val["bridge1"] for _, val in self.Lk.items()]
            b2 = [val["bridge2"] for _, val in self.Lk.items()]
            b3 = [val["bridge3"] for _, val in self.Lk.items()]
            arr = np.array([b1, b2, b3])
            header = [
                f"{int(np.mean(self.magnetic_field[sweep]) / 1e4)}T"
                for sweep in self.magnetic_field.keys()
            ]
            np.savetxt(
                os.path.join(
                    self._savepath, "ppms_analysis", f"ppms_Lk_{self._fname}.txt"
                ),
                arr,
                header="\t".join(header),
                delimiter="\t",
            )
        return self.Lk

    def plot_resist_vs_temp(
        self,
        R_unit: Literal["ohm", "kohm", "Mohm"] = "ohm",
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        title: Optional[str] = None,
        size: tuple | Literal["default"] = "default",
        legend_loc: tuple | str = "best",
        legend_cols: int = 1,
        figure_style: str = "default",
        save: bool = False,
        image_type: str = "svg",
    ) -> Figure:
        """
        Plots the resistance as a function of temperature as measured by the PPMS.

        Parameters
        ----------
        R_unit : {'ohm', 'kohm', 'Mohm'}
        """
        if not os.path.exists(os.path.join(self._savepath, "ppms_analysis")):
            os.mkdir(os.path.join(self._savepath, "ppms_analysis"))
        x_label = "Temperature (K)" if x_label is None else x_label
        y_label = (
            f"Resistance ({R_unit})".replace("ohm", r"$\Omega$")
            if y_label is None
            else y_label
        )
        figures = []
        for bridge in ["bridge1", "bridge2", "bridge3"]:
            resist = [val[bridge] for _, val in self.resistance.items()]
            stddev = [val[bridge] for _, val in self.std_dev.items()]
            figure = gl.Figure(
                x_label=x_label,
                y_label=y_label,
                title=title,
                size=size,
                figure_style=figure_style,
            )
            for i, r in enumerate(resist):
                t = self.temperature[f"Sweep {i+1}"]
                mag_field_mean = np.mean(self.magnetic_field["Sweep {}".format(i + 1)])
                if np.round(mag_field_mean / 10, 1) == 0:
                    label = "0 T"
                elif mag_field_mean / 1e4 < 0.999:
                    label = "{} mT".format(int(mag_field_mean / 10))
                else:
                    label = "{} T".format(np.round(mag_field_mean / 1e4, 1))
                if R_unit == "ohm":
                    curve = gl.Curve(t, r, label=label)
                    curve.add_errorbars(y_error=stddev[i])
                elif R_unit == "kohm":
                    curve = gl.Curve(t, r / 1e3, label=label)
                    curve.add_errorbars(y_error=stddev[i] / 1e3)
                elif R_unit == "Mohm":
                    curve = gl.Curve(t, r / 1e6, label=label)
                    curve.add_errorbars(y_error=stddev[i] / 1e6)
                else:
                    raise ValueError(f"Resistance unit {R_unit} unaccepted")
                figure.add_elements(curve)
            figure.set_visual_params(
                color_cycle=list(
                    color_palette("flare_r", n_colors=len(figure._elements))
                )
            )
            figures.append(figure)
            if save:
                figure.save(
                    os.path.join(
                        self._savepath,
                        "ppms_analysis",
                        f"R_vs_T_{self._fname}_{bridge}.{image_type}",
                    ),
                    legend_loc=legend_loc,
                    legend_cols=legend_cols,
                )
            else:
                figure.show(legend_loc=legend_loc, legend_cols=legend_cols)
        return figures
