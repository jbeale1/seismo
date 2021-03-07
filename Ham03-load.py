#!/usr/bin/python3
# ObsPy code to get data from Earthworm waveserver on BBShark+RPi

from obspy.clients.earthworm import Client
from obspy.core.utcdatetime import UTCDateTime

hours = 12          # hours of data to get (must be divisor of 24)
dur = 60*60*hours   # 60 seconds * 60 minutes * hours
tOffset = (0+4)*60*60   # time range set back this many seconds 
saveFile = True         # save out image file to drive

# IP and port of Earthworm Server
client3 = Client("192.168.1.249", 16022) # BBShark - HAM03
r3 = client3.get_availability('*', '*', channel='EHZ')
  
t = UTCDateTime()     # current UTC time right now
# find most recent top of an even Nth hour (24/N per day)
evenHour = hours*int(t.hour / hours)  # on even Nth hours
winEnd = UTCDateTime(t.year,t.month,t.day,evenHour, 0, 0) - tOffset
winStart = winEnd - dur  # start time
# winEnd = winStart + dur + 1 # one extra second beyond 'hours'

pstStart = winStart - 8*60*60  # get Pacific Standard Time from UTC
pstString = str(pstStart)[:13]

print("Requesting " + str(winStart) + " - " + str(winEnd))
startStr = str(winStart)
outName = r3[0][1]+"_"+pstString+".png"

st = client3.get_waveforms(r3[0][0],r3[0][1],r3[0][2],r3[0][3], \
  winStart, winEnd)
# print(st)

# lowpass filtering 
# tr = st.copy()
st.filter('lowpass', freq=2.0, corners=2, zerophase=True)

ptitle = r3[0][0] + " " + r3[0][1] + " " + r3[0][2] + " " + r3[0][3] \
  + "  Start:" + str(winStart)[:16] + "  PST:" + pstString
  
# annotate day plot with large events: events={'min_magnitude': 6.4},
if (saveFile == True):
  st.plot(type='dayplot',size=(1000,1000),right_vertical_labels=True,
    linewidth=1,title=ptitle,outfile=outName) # save file
else:
  st.plot(type='dayplot',size=(1000,1000),right_vertical_labels=True,
    linewidth=0.5,title=ptitle) # display on screen, not save
