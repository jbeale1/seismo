#!/usr/bin/python3

# ObsPy code to get data from Earthworm waveserver on BBShark+RPi
# and call Octave processing script
# 2020-Dec-27 J.Beale

# /home/pi/hammer/obspy/compareTrace.py

from obspy.clients.earthworm import Client
from obspy.core.utcdatetime import UTCDateTime
# from scipy.io import savemat   # for export to Matlab format file
import sys            # for writing file
import numpy as np    # for writing
import subprocess  # to call Octave script

# ----------------------------------------------------------
# writeDat() - write out data into CSV file
# ----------------------------------------------------------
def writeDat(fname, st):
  print("Writing %s\n" % fname)    # DEBUG verbose show filename
  for i, tr in enumerate(st):
    f = open("%s" % fname, "w")
    f.write("%s\n" % (tr.stats.station))
    f.write("# STATION %s\n" % (tr.stats.station))
    f.write("# CHANNEL %s\n" % (tr.stats.channel))
    f.write("# START_TIME %s\n" % (str(winStart)))
    f.write("# SAMP_FREQ %f\n" % (tr.stats.sampling_rate))
    f.write("# NDAT %d\n" % (tr.stats.npts))
    np.savetxt(f, tr.data * calibration, fmt="%6.0f")
    f.close()  

# ------------------------------------------------------------------

hours = 4           # duration of data to get (must be divisor 24)
calibration = 1.0   # multiply data values by this scale factor

dur = 60*60*hours   # 60 seconds * 60 minutes * hours

# IP and port of EW server
client1 = Client("192.168.1.227", 16022) # SHARK
client2 = Client("192.168.1.129", 16022) # SHRK2
r1 = client1.get_availability('*', '*', channel='EHZ')
r2 = client2.get_availability('*', '*', channel='EHZ')
print(r1)
print(r2)
  
t = UTCDateTime()     # current UTC time right now

# most recent top of an even Nth hour (24/N per day)
evenHour = hours*int(t.hour / hours)  # on even Nth hours
winEnd = UTCDateTime(t.year,t.month,t.day,evenHour, 0, 0)
winStart = winEnd - dur  # start time
winEnd = winStart + dur + 1 # one extra second beyond 'hours'

# ==============================================================

print("Requesting " + str(winStart) + " - " + str(winEnd))

startStr = str(winStart)
fname1 = startStr[0:13]+startStr[14:16]+"_"+r1[0][1]+".csv"
fname2 = startStr[0:13]+startStr[14:16]+"_"+r2[0][1]+".csv"
fresult = startStr[0:13]+startStr[14:16]+"_qual.csv"

# Specify: network, station, location, channel, startTime, endTime
st = client1.get_waveforms(r1[0][0],r1[0][1],r1[0][2],r1[0][3], winStart, winEnd)
print(st)
writeDat(fname1, st)

st = client2.get_waveforms(r2[0][0],r2[0][1],r2[0][2],r2[0][3], winStart, winEnd)
print(st)
writeDat(fname2, st)

prog = '/usr/bin/octave'
script = '/home/pi/hammer/obspy/qualityCheck.m'

print("Now calling %s %s %s %s %s\n" % (prog, script, fname1, fname2, fresult))
subprocess.run([prog,script,fname1,fname2,fresult])

