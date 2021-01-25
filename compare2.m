# GNU Octave script
# Read two CSV format files (eg. seismometer records)
# Compute correlation between them

# https://octave.org/doc/v4.2.0/Simple-File-I_002fO.html
# data = dlmread (file, sep, r0, c0)  r0=start row, c0 = start column
#  r0,c0 indexes start counting from 0

pkg load signal                  # for high & low-pass filter


fs = 100;   # signal sample rate (samples per second)

dir="/home/pi/hammer/obspy";  # working directory
fname1 = "2021-01-23T2007_SHARK.csv";
fname2 = "2021-01-23T2007_SHRK2.csv";

f1=[dir "/" fname1];  # create full pathname
f2=[dir "/" fname2];  # create full pathname

d1 = dlmread(f1, ",", 6,0);  # read CSV file into variable
hours1 = (length(d1)-1) / (fs*60*60);
d2 = dlmread(f2, ",", 6,0);  # read CSV file into variable
hours2 = (length(d2)-1) / (fs*60*60);

printf("%s hours: %5.2f\n", fname1,hours1);
printf("%s hours: %5.2f\n", fname2,hours2);


# plot(d1); hold on; plot(d2);
wlen = 50000;   # signal window length
#startOff = 1632500 + 0*wlen/4;  # starting offset index for S wave
startOff = 16325 + 0*wlen/4;  # start on background microseism

w1=d1(startOff:startOff+wlen);
w2=d2(startOff:startOff+wlen);

fhp = 0.035;    # filter: highpass frequency shoulder in Hz
flp = 0.200;    # lowpass frequency
poles = 2;     # filter poles

[b,a] = butter(poles,double(flp)/fs, "low");  # create lowpass Btwth filter
w1f = filter(b,a, w1);
w2f = filter(b,a, w2);

[b,a] = butter(poles,double(fhp)/fs, "high");  # create highpass Btwth filter
w1f = filter(b,a, w1f);
w2f = filter(b,a, w2f);

# linear correlation 
# AKA product-moment coefficient of correlation, or Pearson's correlation
c = corrcoef(w1f,w2f)(1,2); 

#hold off; subplot(1,1,1);
# plot(w1f); hold on; plot(w2f); grid on; axis("tight");
# ------------------------

r1 = rms(w1f);
r2 = rms(w2f);
scalefac = r1/r2;
w2fs = w2f * scalefac;  # adjusted to match amplitude

# hold off;
# plot(w1f); hold on; plot(w2fs); grid on; axis("tight");

residual = w1f - w2fs;
# c = w1f \ w2fs;    # linear correlation
r1m = rms(residual);

sig1m = rms(w1f);
rs = sig1m / r1m; # rs =  7.2480
res = rs * residual;

[R,lag] = xcorr(w1f, res);
[xpk, xipk] = max(R);  # xipk = 49574   (autocor: 50001) delta=427


offset = (length(res) - xipk);

res1 = res;
res1(1:end+1-offset) = res(offset:end);  # shift waveform up in time

res2 = w1f - res1;
r2m = rms(res2)/rs;
SNR = (sig1m / r2m);
SNRdB = 20 * log(SNR);

w12f = (w1f + w2fs)/2;    # average of both signals
[Pxx, w] = periodogram (w12f);  # power spectral density (real)
plen = length(Pxx);
fmax = fs/2;
xf = linspace((fmax/plen),fmax,plen);  # frequency plot
Pxx = sqrt(Pxx);
[fpeak, ifpeak] = max(Pxx);  # find peak of frequency spectrum
fmax = xf(ifpeak);  # frequency at peak

printf("Start: %5.1f (s) Length: %5.1f (s)\n", (startOff/fs),(wlen/fs));
printf("Correlation: %6.4f\n", c);  # this is the (scalar) correlation value
printf("Fpeak = %5.3f Hz (%5.2f s) %5.2e\n", fmax,(1/fmax),fpeak);   
printf("BP Filter: %5.3f - %5.3f Hz (%d poles)\n", fhp,flp,poles);
printf("      Signal RMS = %5.3f\n", sig1m);
printf("1st residual RMS = %5.3f\n", r1m);
printf("2nd residual RMS = %5.3f (off: %d)\n", r2m,offset);
printf("        SNR (dB) = %5.2f\n", SNRdB);

loglog(xf,Pxx);  # log-log plot of power spectrum
# axis([0.01 0.5 1 200])

# hold off; plot(res1); hold on; grid on; axis("tight"); plot(w1f);
# Plot both channels, scaled to best RMS amplitude match:
# plot(w1f); hold on; plot(w2fs); grid on; axis("tight"); hold off;

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

        
#}
