#!/usr/bin/env python3
import math
import time
import numpy as np
from decimal import Decimal
from astropy import coordinates, units
import sigpyproc.readers as readers

secperday = 3600 * 24


class FilReader:
    def __init__(self, file):
        self.filename = file
        self.device = 0
        self.dm = None
        self.fil = readers.FilReader(file)
        header = self.fil.header

        # get the data
        self.data_type = header.data_type
        self.nchans = header.nchans
        self.foff = header.foff
        self.fch1 = header.fch1
        self.nbits = header.nbits
        self.tsamp = header.tsamp
        self.tstart = header.tstart
        self.nsamples = header.nsamples
        self.nifs = header.nifs
        self.coord = header.coord
        self.azimuth = header.azimuth
        self.zenith = header.zenith
        self.telescope = header.telescope
        self.backend = header.backend
        self.source = header.source
        self.frame = header.frame
        self.ibeam = header.ibeam
        self.nbeams = header.nbeams
        self.dm = header.dm
        self.period = header.period
        self.accel = header.accel
        self.signed = header.signed
        self.ra_deg = self.coord.ra.value
        self.dec_deg = self.coord.dec.value
        self.ra = self.coord.ra.to_string(unit=units.hour, sep=':')
        self.dec = self.coord.dec.to_string(unit=units.degree, sep=':')
        self.chan_freqs = np.arange(self.fch1, self.fch1 + self.foff * self.nchans, self.foff)
        if self.foff < 0:
            self.chan_freqs = np.flip(self.chan_freqs)

        self.freq = (self.chan_freqs[0] + self.chan_freqs[-1] )/2

        # FITS文件总时长(seconds)
        self.total_time_seconds = self.nsamples * self.tsamp
        self.resolution_per_second = int(round(1 / self.tsamp))

    def read_data(self, start_time, end_time):
        delta_time = end_time - start_time

        start_sample = int(self.resolution_per_second * start_time)
        delta_sample = int(self.resolution_per_second * delta_time)
        if delta_sample > self.nsamples - start_sample:
            delta_sample = self.nsamples - start_sample

        data = self.fil.read_block(start_sample, delta_sample).data.astype(np.uint8)
        if self.foff < 0:
            # 因为随着通道号增加，频率下降，所以这里要行（频率通道）反转
            data = data[::-1]

        # fil文件中第一维是时间采样，第二维是频率通道。但是read_block会做转置，所以要再转置回来
        return data.T

    def close(self):
        self.fil = None

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.close()
