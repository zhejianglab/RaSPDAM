#!/usr/bin/env python3
import time
import numpy as np
from decimal import Decimal
import astropy.io.fits as pyfits
from astropy import coordinates, units

secperday = 3600 * 24


class FitsReader:
    def __init__(self, file):
        self.filename = file

    def readFAST(self):
        """
        Help:
        self.filename
        self.device
        self.dm
        self.fits
        self.data1
        self.chan_freqs
        self.dat_scl
        self.nline
        self.nsblk
        self.tbin
        self.npol
        self.nsuboffs
        self.tsamp
        self.chan_bw
        self.freq
        self.nchan
        self.obsbw
        self.telescope
        self.backend
        self.nbits
        self.poln_order
        self.beam
        self.STT_IMJD
        self.STT_SMJD
        self.STT_OFFS
        self.tstart

        self.ra_deg
        self.dec_deg
        self.track_mode
        """

        filename = self.filename
        self.device = 0
        self.dm = None
        self.fits = pyfits.open(filename, mode="readonly", memmap=True, lazy_load_hdus=True)
        hdu0 = self.fits[0]
        hdu1 = self.fits[1]
        # data0 = hdu0.data
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

        return self

    def getdata(self):
        hdu1 = self.fits[1]

        a, b, c, d, e = hdu1.data['DATA'].shape

        numChannel = 4096
        if c > 1:
            dataPol0 = hdu1.data['DATA'][:, :, 0, :, :].squeeze().reshape((-1, numChannel))
            dataPol1 = hdu1.data['DATA'][:, :, 1, :, :].squeeze().reshape((-1, numChannel))
            Scale = np.mean(dataPol0) / np.mean(dataPol1)
            return (dataPol0 + Scale * dataPol1) / 2.

        return hdu1.data['DATA'].squeeze().reshape((-1, numChannel))

    def close(self):
        self.fits.close()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.close()
