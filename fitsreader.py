#!/usr/bin/env python3
import math
import time
import numpy as np
from decimal import Decimal
import astropy.io.fits as pyfits
from astropy import coordinates, units

secperday = 3600 * 24


class FitsReader:
    def __init__(self, file):
        self.filename = file
        self.device = 0
        self.dm = None
        self.fits = pyfits.open(file, mode="readonly", memmap=True, lazy_load_hdus=True)
        hdu0 = self.fits[0]
        hdu1 = self.fits[1]
        data1 = hdu1.data
        header0 = hdu0.header
        header1 = hdu1.header

        # get the data
        # self.data1 = hdu1.data
        self.chan_freqs = self.fits['SUBINT'].data[0]['DAT_FREQ']
        self.dat_scl = np.array(data1['DAT_SCL'])
        self.nline = header1['NAXIS2']
        self.nsblk = header1['NSBLK']
        # self.tbin = header1['TBIN']
        self.npol = header1['NPOL']
        self.nsuboffs = header1['NSUBOFFS']
        self.tsamp = header1['TBIN']
        self.chan_bw = header1['CHAN_BW']
        self.freq = header0['OBSFREQ']
        self.nchan = int(header0['OBSNCHAN'])
        self.obsbw = header0['OBSBW']
        self.telescope = header0["TELESCOP"].strip()
        self.backend = header0["BACKEND"].strip()
        self.nbits = header1["NBITS"]
        self.poln_order = header1["POL_TYPE"]
        self.beam = header0['IBEAM']
        self.STT_IMJD = header0['STT_IMJD']
        self.STT_SMJD = header0['STT_SMJD']
        self.STT_OFFS = header0['STT_OFFS']
        self.tstart = "%.13f" % (Decimal(self.STT_IMJD) + (Decimal(self.STT_SMJD) + Decimal(self.STT_OFFS)) / secperday)

        loc = coordinates.SkyCoord(header0["RA"], header0["DEC"], unit=(units.hourangle, units.deg))
        self.ra_deg = loc.ra.value
        self.dec_deg = loc.dec.value
        self.ra = loc.ra.to_string(unit=units.hour, sep=':')
        self.dec = loc.dec.to_string(unit=units.degree, sep=':')
        self.track_mode = header0['TRK_MODE']

        # FITS文件总时长(seconds)
        self.total_time_seconds = self.nline * self.nsblk * self.tsamp
        x, y, _, _, _ = hdu1.data['DATA'].shape
        self.resolution_per_second = int(round(x * y / self.total_time_seconds))

    def read_data(self, start_time, end_time):
        delta_time = end_time - start_time
        hdu1 = self.fits[1]

        # x,y 维度需要合并, c为偏振，d为频率，默认4096
        x, y, c, d, _ = hdu1.data['DATA'].shape
        # 最后一维数据一般为空
        fits_data = hdu1.data['DATA'][:, :, :, :, 0]

        start_y = int((self.resolution_per_second * start_time) % y)
        start_x = int(math.floor((self.resolution_per_second * start_time) / y))

        offset_y = int((self.resolution_per_second * delta_time) % y)
        offset_x = int(math.floor((self.resolution_per_second * delta_time) / y))

        delta_data = fits_data[start_x:start_x + offset_x + 1, :, :, :]

        if c > 1:
            # 有偏振情况下，取均值
            delta_data = np.average(delta_data, axis=2)
        else:
            delta_data = delta_data[:, :, 0, :]

        reshape_delta_data = delta_data.reshape(-1, d)

        return reshape_delta_data[start_y:offset_y - y, :]

    # def getdata(self):
    #     hdu1 = self.fits[1]
    #
    #     a, b, c, d, e = hdu1.data['DATA'].shape
    #
    #     numChannel = 4096
    #     if c > 1:
    #         dataPol0 = hdu1.data['DATA'][:, :, 0, :, :].squeeze().reshape((-1, numChannel))
    #         dataPol1 = hdu1.data['DATA'][:, :, 1, :, :].squeeze().reshape((-1, numChannel))
    #         Scale = np.mean(dataPol0) / np.mean(dataPol1)
    #         return (dataPol0 + Scale * dataPol1) / 2.
    #
    #     return hdu1.data['DATA'].squeeze().reshape((-1, numChannel))

    def close(self):
        self.fits.close()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.close()
