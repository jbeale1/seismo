# qualityCheck.m - GNU Octave script 
#
# Read two CSV format files (seismometer records)
# Compute stats & correlation between them, in a sliding window
# Exclude and optionally plot areas of lower correlation
#   which is often some kind of local glitch or failure
#
# Takes three arguments: two input filenames, one output filename
# J.Beale 28-JAN-2021
# -------------------------------------------------------------------

pkg load signal                  # for high & low-pass filter

# pkg load optim       # linear regression
# -------------------------------------------------------------------
global startOff  # offset of start of data frame
global stopOff   # offset of end of data frame
global d1        # 1st dataset (raw data)
global d2        # 2nd dataset (raw data)
global w1f       # window into filtered dataset 1
global w2fs      # window into filtered dataset 1 (scaled)
global fname1    # filename of dataset 1
global fname2    # filename of dataset 2
global fs        # sample rate (samples per second)
global c         # linear correlation of this frame
global fmax      # frequency of power spectrum max (Hz)
global bumpFrames = 0  # how many frames had glitch (low correlation)
# ===================================================================

fs = 100;           # signal sample rate (samples per second)
wsizeS = 200;       # size of window (seconds) (was 400)
wstepS = wsizeS/4;  # window step size (seconds)
fhp = 0.035;        # filter: highpass frequency shoulder in Hz
flp = 0.200;        # lowpass frequency
poles = 2;          # filter poles
showPlot = false;   # true will save plot PNGs of glitch areas

dir="/home/pi/hammer/obspy";  # working directory

arg_list = argv ();        # command-line inputs to this function
args = length(arg_list);   # how many args?

if (args < 3)
  printf("Usage: quakeCor1 <fname1> <fname2> <fout>\n");
  printf(" filenames of two CSV files to compare, and output file\n");
  exit(1);
else
  fname1 = arg_list{1};      # data file #1
  fname2 = arg_list{2};      # data file #2
  fname3 = arg_list{3};      # output file
endif

# ===================================================================

# ---------------------------------------------
# Create & save comparison plot of raw, filtered signals
# ---------------------------------------------

function doplot()

global startOff
global stopOff
global d1
global d2
global w1f
global w2fs
global fname1
global fname2
global fs
global c
global fmax

 subplot(1,1,1); 
 hold off;
 set(0,'DefaultTextInterpreter','none'); # avoid TeX markup 
 subplot(2,1,1);
 plot(d1(startOff:stopOff)); axis("tight"); hold on; grid on;
 plot(d2(startOff:stopOff)); 
 yL=ylim(); yR=yL(2)-yL(1); # get Y axis limits, find range
 yP = (0.95*yR) + yL(1); # 90% of the way to the top edge of graph
 yP2 = yL(1) - 0.1*yR; # Just below bottom edge of graph
 yP3 = yL(2) + 0.05*yR; # Just above top edge of graph
 text (200, yP, "SHARK, SHRK2 raw signal");
 text (1, yP2, "sample # (100 Hz, 200 sec)");
 offs = sprintf("+%1.0f",(startOff / fs));
 text (1, yP3, [fname1 "  &  " fname2 "  offset: " offs " s"]);

# ---------------------------------------------
# Show comparison plot of filtered signal
# ---------------------------------------------
 subplot(2,1,2);
 plot(w1f); axis("tight"); hold on; plot(w2fs); grid on;
 yL=ylim(); yR=yL(2)-yL(1); # get Y axis limits, find range
 yP = (0.95*yR) + yL(1); # 90% of the way to the top edge of graph
 yP2 = (0.90*yR) + yL(1); 
 yP3 = (0.85*yR) + yL(1);

 text (200, yP, "BP: 0.035-0.200 Hz");
 legend("SHARK  ", "SHRK2  ", "location", "northeast");

 st2 = sprintf("Corr = %5.4f",c);
 text (200, yP2, st2);
 st3 = sprintf("Fpk = %5.3f Hz",fmax);
 text (200, yP3, st3);

#---------------------------
# save plot image
#---------------------------
 fpout = [fname1(1:end-4) offs ".png"];
 print ("-S2000,900", "-dpng", fpout);

endfunction

# ====================================================================
# keep track of min and max values of a variable
# --------------------------------------------------------------------
function [min max] = pksave(x,xMin,xMax)
  if (x < xMin)
    min = x;
  else
    min = xMin;
  endif
  
  if (x > xMax)
    max = x;
  else
    max = xMax;
  endif
endfunction

# ====================================================================


  # initialize min,max vars with opposite extrema
pMin = 1e9; pMax = 0;    # averaged total RMS power in filtered signal
cMin = 1.0; cMax = -1.0; # correlation 
sMin = 1000; sMax = -1000; # SNR value (dB)
rMin = 1e9;  rMax = 0; # 1st residual of linear fit
fMin = 1e9;  fMax = 0; # frequency of power spectrum maximum
  
f1=[dir "/" fname1];  # create full pathname
f2=[dir "/" fname2];  # create full pathname
f3=[dir "/" fname3];  # create full pathname

d1 = dlmread(f1, ",", 6,0);  # read CSV file into variable
d2 = dlmread(f2, ",", 6,0);  # read CSV file into variable

fout = fopen (f3, "w");  # open output file

d1L = length(d1);
hours1 = (d1L-1) / (fs*60*60);

d2L = length(d2);
hours2 = (d2L-1) / (fs*60*60);

d12L = min(d1L,d2L);

fprintf(fout,"index, sec, corr, fmax, SNR, shift, Res2, Res1\n"); # CSV file header
fprintf(fout,"# %s hours: %5.2f\n", fname1,hours1);
fprintf(fout,"# %s hours: %5.2f\n", fname2,hours2);
if (d1L != d2L)
  fprintf(fout,"# WARNING input channels are not same length\n");
endif

wlen = fs * wsizeS;   # signal window length, in samples
wstep = fs * wstepS;  # window increment, in samples

n = int32(d12L / wstep);
fpArray = zeros(n,1);  # initialize array large enough for all freq vals  
cArray = fpArray;      # initialize others same way
sArray = fpArray;
rArray = fpArray;

Ai = 1; # pointer to current element of fpArray(), cArray() etc.

[b,a] = butter(poles,double(flp)/fs, "low");  # create lowpass
d1f = filter(b,a, d1);
d2f = filter(b,a, d2);

[b,a] = butter(poles,double(fhp)/fs, "high");  # highpass Btwth filter
d1f = filter(b,a, d1f);
d2f = filter(b,a, d2f);
i=0;  # variable used to step window through full dataset


while (true)  
  startOff = 1 + i*wstep;  # start at the beginning
  #startOff = 16080*fs;  # start on background microseism
  stopOff = startOff + wlen;
  if (stopOff > d12L)  # have we reached the end of the data?
    startOff = 1 + (i-1)*wstep;  # step back one
    stopOff = startOff + wlen;  
    break;
  end
  
  w1f=d1f(startOff:stopOff);  # select window into full dataset
  w2f=d2f(startOff:stopOff);

  # linear correlation or product-moment coefficient of correlation
  c = corrcoef(w1f,w2f)(1,2); 

  r1 = rms(w1f);
  r2 = rms(w2f);
  scalefac = r1/r2;
  #scalefac = 1.11;
  
  w2fs = w2f * scalefac;  # adjusted to match amplitude
  residual = w1f - w2fs;  # residue after 1st order linear fit
  average = (w1f + w2fs)/2;  # residue after 1st order linear fit
  
  # c = w1f \ w2fs;    # linear correlation?
  r1m = rms(residual);
  r3 = rms(average);

  #sig1m = rms(w1f);
  rs = r1 / r1m; # rs =  7.2480
  res = rs * residual;  # res now scaled up by (rs)

  [R,lag] = xcorr(w1f, res);
  [xpk, xipk] = max(R);  # xipk = 49574   (autocor: 50001) delta=427

  offset = (length(res) - xipk);
  if (offset < 1)  # probably some noise spike
    offset = 1;
  endif
  # offset usually around 180 on microseism (1.8 seconds)
  
  res1 = res;
  res1(1:end+1-offset) = res(offset:end);  # shift waveform up in time

  #res2 = w1f - res1;
  res2 = (w2fs - res1)/rs; # res2 in same scaling as w1f,w2fs,residual
  r2m = rms(res2);
  SNR = (r1 / r2m);
  SNRdB = 20 * log(SNR);

  w12f = (w1f + w2fs)/2;    # average of both signals
  [Pxx, w] = periodogram (w12f);  # power spectral density (real)
  plen = length(Pxx);
  fN = fs/2;     # Nyquist frequency = sample rate / 2  
  xf = linspace((fN/plen),fN,plen);  # frequency plot
  Pxx = sqrt(Pxx);
  [fpeak, ifpeak] = max(Pxx);  # find peak of frequency spectrum
  fPk = xf(ifpeak);  # frequency at peak

  res2p = r2m*r1m;
  fprintf(fout,"%03d, %5.0f, %6.4f, %5.3f, %5.2f, %d, %5.4f, %5.4f\n",
     i,(startOff/fs),c,fPk,SNRdB,offset,res2p,r1m);
  if (res2p > 4000)  # Whoops! difference too large; something wrong
    if (showPlot)
      doplot()  # generate & save out the plot
    endif
    bumpFrames = bumpFrames + 1;
  else
    # good frame: update records if this was a min or max value
    [pMin pMax] = pksave(r3, pMin, pMax);  # avg RMS min & max
    [cMin cMax] = pksave(c, cMin, cMax);  # correlation min & max
    [sMin sMax] = pksave(SNRdB, sMin, sMax); # SNR min & max
    [rMin rMax] = pksave(r1m, rMin, rMax); # 1st Residual min & max
    [fMin fMax] = pksave(fPk, fMin, fMax); # 1st Residual min & max
    pArray(Ai) = r3;     # record each correlation coeff.
    cArray(Ai) = c;     # record each correlation coeff.
    sArray(Ai) = SNRdB;
    rArray(Ai) = r1m;
    fpArray(Ai) = fPk;  # record each peak frequency
    Ai = Ai + 1;
  endif
   
  i += 1;
endwhile

# =====================================================================
#       sort all peak frequency readings, to enable finding median
pAS  = sort(pArray(1:Ai-1));
cAS  = sort(cArray(1:Ai-1));
sAS  = sort(sArray(1:Ai-1));
rAS  = sort(rArray(1:Ai-1));
fpAS = sort(fpArray(1:Ai-1));

mpoint = int32(Ai / 2);  # off-by-1/2 if Ai is even, but oh well
m2 = int32(Ai / 3);
m3 = m2*2;
# find the median 
pMedian = pAS(mpoint);
cMedian = cAS(mpoint);
sMedian = sAS(mpoint);
rMedian = rAS(mpoint);
fMedian = fpAS(mpoint);  

# find the width of the central 1/3 of the full distribution
pT = pAS(m3) - pAS(m2);
cT = cAS(m3) - cAS(m2);
sT = sAS(m3) - sAS(m2);
rT = rAS(m3) - rAS(m2);
fT = fpAS(m3) - fpAS(m2);

hCover = ((wstepS * i) + (wsizeS - wstepS)); # seconds covered
fprintf(fout,"# RMS:  %5.3f %5.3f %5.3f (%5.3f)\n",pMin,pMedian,pMax,pT);
fprintf(fout,"# Res1: %5.3f %5.3f %5.3f (%5.3f)\n",rMin,rMedian,rMax,rT);
fprintf(fout,"# Corr: %5.3f %5.3f %5.3f (%5.3f)\n",cMin,cMedian,cMax,cT);
fprintf(fout,"# SNR:  %5.3f %5.3f %5.3f (%5.3f)\n",sMin,sMedian,sMax,sT);
fprintf(fout,"# Fpk:  %5.3f %5.3f %5.3f (%5.3f)\n",fMin,fMedian,fMax,fT);
fprintf(fout,"# Good frames: %d / %d\n",Ai-1,(Ai+bumpFrames-1));
fprintf(fout,"# Bump frames: %d / %d  (%5.3f hours)\n",
    bumpFrames,i,hCover/3600);

# --------------
#{

-----------------------------------
Input Files:
-----------------------------------
2021-01-23T2007_SHARK.csv hours:  6.00
2021-01-23T2007_SHRK2.csv hours:  6.00

-----------------------------------
During quake signal:
-----------------------------------
Start: 16325.0 (s) Length: 500.0 (s)
Correlation: 0.9911
Fpeak = 0.066 Hz (15.24 s) 3.02e+04
BP Filter: 0.035 - 0.200 Hz (2 poles)
      Signal RMS = 619.634
1st residual RMS = 82.489
2nd residual RMS = 36.606 (off: 399)
        SNR (dB) = 56.58
        
------------------------------------
During background:
------------------------------------
Start: 163.2 (s) Length: 500.0 (s)
Correlation: 0.8224
Fpeak = 0.146 Hz ( 6.83 s) 8.57e+02
BP Filter: 0.035 - 0.200 Hz (2 poles)
      Signal RMS = 30.016
1st residual RMS = 17.893
2nd residual RMS = 18.404 (off: 128)
        SNR (dB) =  9.78    

Correlation: 0.8224
Fpeak = 0.146 Hz (6.83 s) 8.57e+02
Signal RMS = 30.016  SNR = 9.78 dB
---
Overall 6 hour file:
# Correlation min/max: -0.386 0.994
# SNR min/max: -21.701 56.257
# Residual min/max: 12.033 135.208

        
#}
