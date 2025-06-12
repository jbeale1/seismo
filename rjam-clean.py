# raspberryJam-v4 seismometer input has 3x MAX11200 24-bit ADC devices
# apparently susceptible to constant-amplitude interference near 1 Hz
# This code uses hand-tuned parameters and PLL tracking to remove it
# J.Beale 11-June-2025 (with AI assistance)

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from obspy import read, UTCDateTime, Trace
import warnings

fInt = None # Global variable to store interference frequency

def detect_interference_frequency(data, fs, freq_range=(0.5, 1.5), min_prominence=2.0, 
                                window_length=8192, overlap=0.75, min_windows=10):
    """
    Automatically detect the dominant interference frequency using averaged power spectral density
    
    Parameters:
    data: input seismic data
    fs: sampling frequency
    freq_range: tuple of (min_freq, max_freq) to search in Hz
    min_prominence: minimum prominence (in dB) for peak detection
    window_length: FFT window length (samples)
    overlap: overlap fraction between windows (0-1)
    min_windows: minimum number of windows required for reliable statistics
    
    Returns:
    detected_freq: frequency of strongest peak in range
    peak_power: power at detected frequency
    snr_db: signal-to-noise ratio in dB
    bandwidth_3db: -3dB bandwidth around peak
    """
    
    print(f"Detecting interference frequency in range {freq_range[0]:.1f} - {freq_range[1]:.1f} Hz")
    
    # Calculate window parameters
    window_length = min(window_length, len(data)//4)  # Ensure reasonable window size
    step_size = int(window_length * (1 - overlap))
    
    # Calculate number of windows
    n_windows = (len(data) - window_length) // step_size + 1
    
    if n_windows < min_windows:
        print(f"Warning: Only {n_windows} windows available, reducing window size for better statistics")
        window_length = len(data) // (min_windows + 2)
        step_size = int(window_length * (1 - overlap))
        n_windows = (len(data) - window_length) // step_size + 1
    
    print(f"Using {n_windows} overlapping windows of {window_length} samples each ({overlap*100:.0f}% overlap)")
    
    # Initialize arrays for accumulating PSDs
    f = None
    psd_accumulator = None
    valid_windows = 0
    
    # Process each window
    for i in range(n_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_length
        
        if end_idx > len(data):
            break
            
        # Extract window and apply Hann taper
        window_data = data[start_idx:end_idx]
        window_data = window_data * signal.windows.hann(len(window_data))
        
        # Compute PSD for this window
        f_win, psd_win = signal.periodogram(window_data, fs, window='boxcar', 
                                          detrend='linear', scaling='density')
        
        # Initialize accumulator on first window
        if f is None:
            f = f_win
            psd_accumulator = np.zeros_like(psd_win)
        
        # Accumulate PSD
        psd_accumulator += psd_win
        valid_windows += 1
    
    # Average the accumulated PSDs
    psd_avg = psd_accumulator / valid_windows
    
    print(f"Averaged {valid_windows} windows for robust spectral estimate")
    
    # Convert to dB scale for peak detection
    psd_db = 10 * np.log10(psd_avg + 1e-12)
    
    # Find frequency indices within our search range
    freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
    f_search = f[freq_mask]
    psd_search = psd_avg[freq_mask]
    psd_db_search = psd_db[freq_mask]
    
    if len(f_search) < 10:
        raise ValueError(f"Insufficient frequency resolution in range {freq_range}")
    
    # Find peaks with minimum prominence
    peak_indices, peak_properties = signal.find_peaks(
        psd_db_search, 
        prominence=min_prominence,
        distance=int(0.05 * len(psd_search))  # Minimum 0.05 Hz separation
    )
    
    if len(peak_indices) == 0:
        # No prominent peaks found, use maximum value
        max_idx = np.argmax(psd_search)
        detected_freq = f_search[max_idx]
        peak_power = psd_search[max_idx]
        print(f"  No prominent peaks found, using maximum at {detected_freq:.3f} Hz")
    else:
        # Find the strongest peak
        peak_powers = psd_search[peak_indices]
        strongest_peak_idx = peak_indices[np.argmax(peak_powers)]
        detected_freq = f_search[strongest_peak_idx]
        peak_power = psd_search[strongest_peak_idx]
        
        print(f"  Found {len(peak_indices)} prominent peaks")
        print(f"  Strongest peak at {detected_freq:.3f} Hz (power: {peak_power:.2e})")
    
    # Calculate broadband noise floor and SNR
    # Use frequencies outside the search range to estimate noise floor
    noise_mask = (f < freq_range[0] - 0.2) | (f > freq_range[1] + 0.2)
    noise_mask = noise_mask & (f > 0.1) & (f < min(10.0, fs/2))  # Reasonable frequency range
    
    if np.sum(noise_mask) > 10:
        noise_floor = np.median(psd_avg[noise_mask])  # Use median for robustness
        noise_floor_db = 10 * np.log10(noise_floor)
    else:
        # Fallback: use local noise estimate
        target_idx = np.argmin(np.abs(f - detected_freq))
        search_width = int(0.2 * len(psd_avg) / (f[-1] - f[0]))  # 0.2 Hz equivalent
        
        noise_start = max(0, target_idx - search_width)
        noise_end = min(len(psd_avg), target_idx + search_width)
        
        noise_bins = np.concatenate([
            psd_avg[noise_start:max(0, target_idx-5)],
            psd_avg[min(len(psd_avg), target_idx+6):noise_end]
        ])
        
        if len(noise_bins) > 0:
            noise_floor = np.median(noise_bins)
            noise_floor_db = 10 * np.log10(noise_floor)
        else:
            noise_floor = peak_power * 0.1  # Fallback estimate
            noise_floor_db = 10 * np.log10(noise_floor)
    
    # Calculate SNR
    peak_power_db = 10 * np.log10(peak_power)
    snr_db = peak_power_db - noise_floor_db
    snr_ratio = peak_power / noise_floor
    
    print(f"  Noise floor: {noise_floor_db:.1f} dB")
    print(f"  Peak power: {peak_power_db:.1f} dB")
    print(f"  SNR: {snr_db:.1f} dB (ratio: {snr_ratio:.1f})")
    
    # Calculate -3dB bandwidth
    target_idx_full = np.argmin(np.abs(f - detected_freq))
    peak_power_db_full = 10 * np.log10(psd_avg[target_idx_full])
    threshold_db = peak_power_db_full - 3.0  # -3dB point
    
    # Find frequencies where power drops below -3dB threshold
    psd_db_full = 10 * np.log10(psd_avg + 1e-12)
    
    # Search around the peak for -3dB points
    search_range = int(0.5 / (f[1] - f[0]))  # Search ±0.5 Hz around peak
    start_search = max(0, target_idx_full - search_range)
    end_search = min(len(f), target_idx_full + search_range)
    
    # Find -3dB points on both sides of peak
    left_idx = target_idx_full
    right_idx = target_idx_full
    
    # Search left side
    for i in range(target_idx_full, start_search, -1):
        if psd_db_full[i] < threshold_db:
            left_idx = i
            break
    
    # Search right side  
    for i in range(target_idx_full, end_search):
        if psd_db_full[i] < threshold_db:
            right_idx = i
            break
    
    bandwidth_3db = f[right_idx] - f[left_idx]
    
    print(f"  -3dB bandwidth: {bandwidth_3db:.4f} Hz ({f[left_idx]:.3f} - {f[right_idx]:.3f} Hz)")
    
    # Enhanced plotting with multiple subplots (FIXED: start at 0.1 Hz)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Full spectrum - START AT 0.1 Hz to avoid log(0) issues
    f_plot_mask = f >= 0.1
    axes[0,0].semilogy(f[f_plot_mask], psd_avg[f_plot_mask], 'b-', alpha=0.7, linewidth=1)
    axes[0,0].axvline(detected_freq, color='r', linestyle='--', linewidth=2, 
                      label=f'Detected: {detected_freq:.3f} Hz')
    axes[0,0].axhline(noise_floor, color='orange', linestyle=':', linewidth=2,
                      label=f'Noise floor: {noise_floor_db:.1f} dB')
    axes[0,0].axvspan(freq_range[0], freq_range[1], alpha=0.2, color='yellow', 
                      label=f'Search range')
    axes[0,0].set_xlabel('Frequency (Hz)')
    axes[0,0].set_ylabel('Power Spectral Density')
    axes[0,0].set_title('Full Spectrum (Averaged)')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend()
    axes[0,0].set_xlim(0.1, min(5, fs/2))  # Start at 0.1 Hz
    
    # Search range detail
    axes[0,1].semilogy(f_search, psd_search, 'b-', alpha=0.7, linewidth=1.5)
    axes[0,1].axvline(detected_freq, color='r', linestyle='--', linewidth=2,
                      label=f'Peak: {detected_freq:.3f} Hz')
    axes[0,1].axhline(noise_floor, color='orange', linestyle=':', linewidth=2,
                      label=f'Noise floor')
    if len(peak_indices) > 0:
        axes[0,1].scatter(f_search[peak_indices], psd_search[peak_indices], 
                         color='red', s=50, zorder=5, label='Detected peaks')
    axes[0,1].set_xlabel('Frequency (Hz)')
    axes[0,1].set_ylabel('Power Spectral Density')
    axes[0,1].set_title(f'Search Range Detail ({freq_range[0]}-{freq_range[1]} Hz)')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].legend()
    
    # dB scale around peak for bandwidth measurement
    zoom_range = 0.2
    zoom_mask = (f >= detected_freq - zoom_range) & (f <= detected_freq + zoom_range)
    if np.any(zoom_mask):
        axes[1,0].plot(f[zoom_mask], psd_db_full[zoom_mask], 'b-', linewidth=1.5)
        axes[1,0].axvline(detected_freq, color='r', linestyle='--', linewidth=2,
                         label=f'Peak: {detected_freq:.3f} Hz')
        axes[1,0].axhline(threshold_db, color='orange', linestyle=':', linewidth=2,
                         label='-3dB threshold')
        axes[1,0].axvline(f[left_idx], color='green', linestyle=':', alpha=0.7, label='-3dB points')
        axes[1,0].axvline(f[right_idx], color='green', linestyle=':', alpha=0.7)
        axes[1,0].set_xlabel('Frequency (Hz)')
        axes[1,0].set_ylabel('Power (dB)')
        axes[1,0].set_title(f'Bandwidth Measurement\nBW = {bandwidth_3db:.4f} Hz')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].legend()
    
    # SNR visualization
    axes[1,1].bar(['Noise Floor', 'Peak Power'], 
                  [noise_floor_db, peak_power_db],
                  color=['orange', 'red'], alpha=0.7)
    axes[1,1].set_ylabel('Power (dB)')
    axes[1,1].set_title(f'SNR Analysis\nSNR = {snr_db:.1f} dB')
    axes[1,1].grid(True, alpha=0.3)
    
    # Add SNR annotation
    axes[1,1].annotate(f'{snr_db:.1f} dB', 
                       xy=(0.5, (noise_floor_db + peak_power_db)/2),
                       xytext=(0.7, (noise_floor_db + peak_power_db)/2),
                       arrowprops=dict(arrowstyle='<->', color='black'),
                       fontsize=12, ha='center')
    
    plt.tight_layout()
    plt.show()
    
    return detected_freq, peak_power, snr_db, bandwidth_3db


def simple_continuous_tracking(data, fs, f_target=1.0, window_length=8.0, update_interval=0.5,
                           delta_f_external_init=0.0, f_loop_bw_low=0.0015, 
                           track_interval_sec=10.0, alpha_update=0.2, amp_alpha=0.5):
    """
    Real-time PLL with slow Δf tracker and amplitude tracking
    Added amp_alpha parameter for amplitude tracking loop
    """
    n_samples = len(data)
    n_window = int(window_length * fs)
    n_update = int(update_interval * fs)
    n_blocks = (n_samples - n_window) // n_update

    # Loop gains
    zeta = 0.707
    wn = 2 * np.pi * f_loop_bw_low
    Kp = 2 * zeta * wn
    Ki = wn ** 2
    print(f"Loop gains: Kp={Kp:.4f}, Ki={Ki:.6f}")

    # harmonic terms relative to fundamental
    h2_amp_ratio = 0.02    # relative to fundamental amplitude
    h2_phase_offset = -np.pi/2.0 # phase offset in radians

    h3_amp_ratio = 0.1    # relative to fundamental amplitude
    h3_phase_offset = 0.0 # phase offset in radians

    h5_amp_ratio = 0.03    # relative to fundamental
    h5_phase_offset = 0.0 # phase offset in radians

    # PLL state
    freq_error = 0.0
    phase_accumulator = 0.0
    delta_f_external = delta_f_external_init

    # Phase unwrap history buffer
    phase_history_buffer = []
    time_history_buffer = []
    delta_f_external_history = []

    # Outputs
    amp_hist = []
    phase_hist = []
    phase_err_wrapped_hist = []
    phase_err_unwrapped_hist = []
    time_points = []
    cleaned_data = np.zeros_like(data)
    subtracted_signal = np.zeros_like(data)


    # Add amplitude tracking variables
    amp_tracking = None
    amp_history = []
    
    # Initialize peak tracking buffer
    peak_buffer_len = int(1.0 * fs)  # 1 second buffer
    peak_buffer = []
    
    # Define when to freeze initial amplitude tracking
    freeze_after_N_blocks = int(1.0 / update_interval)  # freeze after N seconds
    
    # Processing loop
    for i_block in range(n_blocks):
        i_start = i_block * n_update
        i_end = i_start + n_window
        if i_end > n_samples:
            break

        x_block = data[i_start:i_end]

        # Generate VCO reference
        t_block = np.arange(n_window) / fs
        vco_freq = f_target + delta_f_external + freq_error
        phase_block = 2 * np.pi * vco_freq * t_block + phase_accumulator
        ref_I = np.cos(phase_block)
        ref_Q = np.sin(phase_block)


        # Mix
        mix_I = np.dot(x_block, ref_I) / n_window
        mix_Q = np.dot(x_block, ref_Q) / n_window

        inst_amp = np.sqrt(mix_I ** 2 + mix_Q ** 2)
        
        # Track peaks instead of instantaneous amplitude
        peak_buffer.append(inst_amp)
        if len(peak_buffer) > peak_buffer_len:
            peak_buffer.pop(0)
        
        a_scale = 1.98 # deeply unsettling hand-tuned fudge factor
        if amp_tracking is None:
            if i_block >= freeze_after_N_blocks:
                amp_tracking = a_scale * max(peak_buffer)  
            amp_est = inst_amp  # Use instantaneous amplitude until tracking starts
        else:
            # Track the peak amplitude with bias toward larger values
            peak_amp = max(peak_buffer)
            amp_error = a_scale * peak_amp - amp_tracking  # Scale up peak estimate
            if amp_error > 0:  # Respond faster to increases
                amp_tracking += 2.0 * amp_alpha * amp_error
            else:  # Respond slower to decreases
                amp_tracking += 0.5 * amp_alpha * amp_error
            amp_est = amp_tracking

        phase_est = np.arctan2(mix_Q, mix_I)
        phase_error = -phase_est  # negative feedback

        # Save history
        time_sec = (i_start + n_window // 2) / fs
        amp_hist.append(amp_est)
        phase_hist.append(phase_est)
        phase_err_wrapped_hist.append(phase_error)
        time_points.append(time_sec)

        # Subtract the reference signal from the input block
        ref_harmonic_block = amp_est * (
              np.cos(phase_block)                                       # Fundamental
            + h2_amp_ratio * np.cos(2 * phase_block + h2_phase_offset)  # 3rd harmonic
            + h3_amp_ratio * np.cos(3 * phase_block + h3_phase_offset)  # 3rd harmonic
            + h5_amp_ratio * np.cos(5 * phase_block + h5_phase_offset)  # 5th harmonic
        )

        cleaned_block = x_block - ref_harmonic_block
        cleaned_data[i_start:i_end] = cleaned_block
        subtracted_signal[i_start:i_end] = ref_harmonic_block


        # === Phase unwrap buffer ===
        if len(phase_history_buffer) == 0:
            phase_unwrapped = phase_error
        else:
            # Unwrap manually based on last value
            delta_phase = phase_error - phase_history_buffer[-1]
            delta_phase = np.mod(delta_phase + np.pi, 2 * np.pi) - np.pi
            phase_unwrapped = phase_history_buffer[-1] + delta_phase

        phase_history_buffer.append(phase_unwrapped)
        time_history_buffer.append(time_sec)

        phase_err_unwrapped_hist.append(phase_unwrapped)

        # === PLL update ===
        freq_error += Ki * phase_error * update_interval
        freq_error = np.clip(freq_error, -0.5, +0.5)  # clip to avoid crazy jumps
        phase_accumulator += 2 * np.pi * (f_target + delta_f_external + freq_error) * n_update / fs

        # === Slow delta-f tracker ===
        # Every N seconds worth of buffer → update delta_f_external
        track_N_blocks = int(track_interval_sec / update_interval)
        if len(phase_history_buffer) >= track_N_blocks:
            # Use only last N sec worth of history
            phase_buf_array = np.array(phase_history_buffer[-track_N_blocks:])
            time_buf_array = np.array(time_history_buffer[-track_N_blocks:])

            A = np.vstack([time_buf_array, np.ones_like(time_buf_array)]).T
            slope, intercept = np.linalg.lstsq(A, phase_buf_array, rcond=None)[0]
            delta_f_est = slope / (2 * np.pi)

            # Update delta_f_external with smoothing
            delta_f_external = (1 - alpha_update) * delta_f_external + alpha_update * delta_f_est
            delta_f_external_history.append((time_sec, delta_f_external))  # update history

            print(f"Block {i_block+1}/{n_blocks}: Δf_est = {delta_f_est * 1000:.3f} mHz, delta_f_external = {delta_f_external * 1000:.3f} mHz")

    # Final output
    return (cleaned_data, subtracted_signal, time_points, amp_hist, phase_hist, phase_err_wrapped_hist, phase_err_unwrapped_hist, delta_f_external_history, amp_history)



def analyze_interference_removal(original, cleaned, subtracted, fs, f_target=1.0):
    """
    Analyze the effectiveness of interference removal
    """
    
    # FFT analysis with better frequency resolution
    n_fft = min(8192, len(original))
    freqs = fftfreq(n_fft, 1/fs)[:n_fft//2]
    
    fft_orig = np.abs(fft(original[:n_fft], n=n_fft))[:n_fft//2]
    fft_clean = np.abs(fft(cleaned[:n_fft], n=n_fft))[:n_fft//2]
    fft_sub = np.abs(fft(subtracted[:n_fft], n=n_fft))[:n_fft//2]
    
    # Find interference frequency bin
    target_bin = np.argmin(np.abs(freqs - f_target))
    
    # Calculate suppression ratio with averaging
    freq_range = 3
    start_bin = max(0, target_bin - freq_range//2)
    end_bin = min(len(fft_orig), target_bin + freq_range//2 + 1)
    
    orig_power = np.mean(fft_orig[start_bin:end_bin])
    clean_power = np.mean(fft_clean[start_bin:end_bin])
    
    suppression_db = 20 * np.log10(orig_power / (clean_power + 1e-12))
    
    print(f"Interference suppression: {suppression_db:.1f} dB at {f_target} Hz")
    print(f"Original power at target: {orig_power:.2e}")
    print(f"Cleaned power at target: {clean_power:.2e}")
    
    # Enhanced plotting
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Time domain comparison
    plot_duration = min(60, len(original)/fs)
    plot_samples = int(plot_duration * fs)
    t = np.arange(plot_samples) / fs;
    
    axes[0,0].plot(t, original[:plot_samples], 'b-', alpha=0.8, label='Original', linewidth=1)
    axes[0,0].plot(t, cleaned[:plot_samples], 'r-', alpha=0.8, label='Cleaned', linewidth=1)
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].set_ylabel('Amplitude')
    axes[0,0].set_title(f'Time Domain Comparison')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Subtracted signal
    axes[0,1].plot(t, subtracted[:plot_samples], 'g-', alpha=0.8, linewidth=1)
    axes[0,1].set_xlabel('Time (s)')
    axes[0,1].set_ylabel('Amplitude')
    axes[0,1].set_title('Subtracted Interference Signal')
    axes[0,1].grid(True, alpha=0.3)
    
    # Difference signal (residual)
    residual = original[:plot_samples] - cleaned[:plot_samples] - subtracted[:plot_samples]
    axes[0,2].plot(t, residual, 'm-', alpha=0.8, linewidth=1)
    axes[0,2].set_xlabel('Time (s)')
    axes[0,2].set_ylabel('Amplitude')
    axes[0,2].set_title('Residual (should be ~zero)')
    axes[0,2].grid(True, alpha=0.3)
    
    # Frequency domain - full spectrum
    axes[1,0].loglog(freqs[1:], fft_orig[1:], 'b-', alpha=0.7, label='Original', linewidth=1)
    axes[1,0].loglog(freqs[1:], fft_clean[1:], 'r-', alpha=0.7, label='Cleaned', linewidth=1)
    # axes[1,0].axvline(f_target, color='k', linestyle='--', alpha=0.5, label='Target freq')
    axes[1,0].set_xlabel('Frequency (Hz)')
    axes[1,0].set_ylabel('Magnitude')
    axes[1,0].set_title('Frequency Domain (Log Scale)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_xlim(0.1, 50)
    
    # Frequency domain - zoomed around interference
    zoom_range = 0.2
    f_zoom = (freqs >= f_target-zoom_range) & (freqs <= f_target+zoom_range)
    if np.any(f_zoom):
        axes[1,1].semilogy(freqs[f_zoom], fft_orig[f_zoom], 'b-', alpha=0.7, 
                          label='Original', linewidth=1.5)
        axes[1,1].semilogy(freqs[f_zoom], fft_clean[f_zoom], 'r-', alpha=0.7, 
                          label='Cleaned', linewidth=1.5)
        axes[1,1].semilogy(freqs[f_zoom], fft_sub[f_zoom], 'g-', alpha=0.7, 
                          label='Subtracted', linewidth=1.5)
        # axes[1,1].axvline(f_target, color='k', linestyle='--', alpha=0.5, label='Target freq')
        axes[1,1].set_xlabel('Frequency (Hz)')
        axes[1,1].set_ylabel('Magnitude')
        axes[1,1].set_title(f'Zoom: {f_target-zoom_range:.1f} - {f_target+zoom_range:.1f} Hz')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    
    # Phase comparison at target frequency
    phase_orig = np.angle(fft(original[:n_fft], n=n_fft))[:n_fft//2]
    phase_clean = np.angle(fft(cleaned[:n_fft], n=n_fft))[:n_fft//2]
    
    if np.any(f_zoom):
        axes[1,2].plot(freqs[f_zoom], np.degrees(phase_orig[f_zoom]), 'b-', 
                      alpha=0.7, label='Original', linewidth=1.5)
        axes[1,2].plot(freqs[f_zoom], np.degrees(phase_clean[f_zoom]), 'r-', 
                      alpha=0.7, label='Cleaned', linewidth=1.5)
        # axes[1,2].axvline(f_target, color='k', linestyle='--', alpha=0.5, label='Target freq')
        axes[1,2].set_xlabel('Frequency (Hz)')
        axes[1,2].set_ylabel('Phase (degrees)')
        axes[1,2].set_title('Phase Comparison')
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return suppression_db

def read_miniseed_data(filename, start_time=None, end_time=None, channel=None, 
                      trace_number=None, merge_method='merge'):
    """Read miniSEED data from file with optional time windowing"""
    try:
        st = read(filename)
        
        if channel is not None:
            st = st.select(channel=channel)
            if len(st) == 0:
                raise ValueError(f"Channel {channel} not found in file")
        
        if len(st) == 0:
            raise ValueError("No traces found in miniSEED file")
        
        if len(st) > 1:
            print(f"Found {len(st)} traces:")
            for i, tr in enumerate(st):
                print(f"  Trace {i+1}: {tr.stats.starttime} to {tr.stats.endtime} "
                      f"({len(tr.data)} samples)")
        
        if start_time is not None:
            if isinstance(start_time, str):
                start_time = UTCDateTime(start_time)
            elif not isinstance(start_time, UTCDateTime):
                start_time = UTCDateTime(start_time)
        
        if end_time is not None:
            if isinstance(end_time, str):
                end_time = UTCDateTime(end_time)
            elif not isinstance(end_time, UTCDateTime):
                end_time = UTCDateTime(end_time)
        
        if trace_number is not None:
            if trace_number < 1 or trace_number > len(st):
                raise ValueError(f"Trace number {trace_number} not valid (file has {len(st)} traces)")
            tr = st[trace_number - 1]
            print(f"Using trace {trace_number}")
            
        elif start_time is not None and merge_method == 'select':
            selected_trace = None
            for i, tr in enumerate(st):
                if tr.stats.starttime <= start_time <= tr.stats.endtime:
                    selected_trace = tr
                    print(f"Auto-selected trace {i+1} containing start time")
                    break
            
            if selected_trace is None:
                raise ValueError(f"No trace contains start time {start_time}")
            tr = selected_trace
            
        elif len(st) > 1 and merge_method == 'merge':
            try:
                st.merge(method=1, fill_value=0)
                tr = st[0]
                print("Merged multiple traces (gaps filled with zeros)")
            except Exception as e:
                print(f"Warning: Could not merge traces ({e}), using first trace")
                tr = st[0]
        else:
            tr = st[0]
            if len(st) > 1:
                print("Using first trace only")
        
        if start_time is not None or end_time is not None:
            tr = tr.slice(starttime=start_time, endtime=end_time)
            
            if len(tr.data) == 0:
                print("Time window not found in selected trace. Available traces:")
                for i, orig_tr in enumerate(st):
                    print(f"  Trace {i+1}: {orig_tr.stats.starttime} to {orig_tr.stats.endtime}")
                raise ValueError(f"No data in specified time range {start_time} to {end_time}")
        
        data = tr.data.astype(np.float64)
        fs = tr.stats.sampling_rate
        actual_start = tr.stats.starttime
        stats = tr.stats
        
        print(f"Loaded {len(data)} samples from {stats.station}.{stats.network}.{stats.channel}")
        print(f"Sampling rate: {fs} Hz")
        print(f"Start time: {actual_start}")
        print(f"End time: {actual_start + len(data)/fs}")
        print(f"Duration: {len(data)/fs:.1f} seconds")
        
        return data, fs, actual_start, stats
        
    except Exception as e:
        print(f"Error reading miniSEED file: {e}")
        raise

def save_cleaned_miniseed(out_data, original_stats, start_time, output_file):
    """Save cleaned data as a new miniSEED file"""
    try:
        from obspy.core import Trace, Stream
        
        new_stats = original_stats.copy()
        new_stats.starttime = start_time
        new_stats.npts = len(out_data)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            tr_new = Trace(data=out_data.astype(np.float32), header=new_stats)
            st_new = Stream([tr_new])
            
            st_new.write(output_file, format='MSEED')
            print(f"Data saved to: {output_file}")
        
    except Exception as e:
        print(f"Error saving miniSEED file: {e}")

def process_miniseed_file_adaptive(filename, start_time=None, end_time=None, 
                                 channel=None, trace_number=None, f_interference=None, 
                                 output_file=None, plot_results=True, merge_method='select',
                                 method='hybrid', freq_range=(0.5, 1.5)):
    """
    - 'adaptive_tracking': Continuous phase/amplitude tracking
    
    Parameters:
    f_interference: interference frequency in Hz. If None, will auto-detect in freq_range
    freq_range: tuple of (min_freq, max_freq) for auto-detection
    """
    global fInt # Use global variable for interference frequency

    # Step 1: Read miniSEED data
    print("Reading miniSEED file...")
    data, fs, actual_start, stats = read_miniseed_data(
        filename, start_time, end_time, channel, trace_number, merge_method
    )
    
    # Step 2: Pre-process data
    print(f"\nPre-processing data...")
    data = signal.detrend(data, type='linear')
    
    if fs > 10:  # highpass to remove drift below 0.1 Hz
        hp_freq = 0.1
        sos = signal.butter(2, hp_freq/(fs/2), btype='high', output='sos')
        data = signal.sosfiltfilt(sos, data)
        print(f"Applied {hp_freq} Hz high-pass filter")
    
    # Step 3: Auto-detect interference frequency if not provided
    if f_interference is None:
        print(f"\nAuto-detecting interference frequency...")
        f_interference, peak_power, snr_db, bandwidth_3db = detect_interference_frequency(
            data, fs, freq_range
        )
        print(f"Auto-detected interference frequency: {f_interference:.3f} Hz")
        print(f"Measured -3dB bandwidth: {bandwidth_3db:.4f} Hz")
        fInt = f_interference
        
        # Check if the detected signal is strong enough to warrant removal
        if snr_db < 3.0:
            print(f"Warning: Detected signal has low SNR ({snr_db:.1f} dB)")
            print("Consider increasing freq_range or checking if interference is present")
    else:
        print(f"Using specified interference frequency: {f_interference:.3f} Hz")
    
    # Step 4: Apply selected method
    print(f"\nApplying {method} interference removal at {f_interference:.3f} Hz...")
    

    if method == 'simple_continuous_tracking':

        # for N signal, f = 0.950
        cleaned_data, subtracted_signal, time_points, amp_hist, phase_hist, phase_err_wrapped_hist, phase_err_unwrapped_hist, delta_f_external_history, amp_history = \
    simple_continuous_tracking(data, fs,
                               f_target=f_interference,
                               window_length=8.0,
                               update_interval=0.5,
                               delta_f_external_init=0.0,
                               f_loop_bw_low=0.0175,
                               track_interval_sec=4,
                               alpha_update=0.05) # was 0.35


        # === Plot Phase Error Unwrap ===
        plt.figure(figsize=(10, 4))
        plt.plot(time_points, phase_err_unwrapped_hist, label="Phase error (unwrapped)", linewidth=1.5)
        plt.xlabel("Time (s)")
        plt.ylabel("Phase error (rad)")
        plt.title("Phase Error Unwrapped vs Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # === Plot delta_f_external vs Time ===
        # Unpack delta_f_external_history
        deltaf_times, deltaf_values = zip(*delta_f_external_history)

        plt.figure(figsize=(10, 4))
        plt.plot(deltaf_times, np.array(deltaf_values) * 1000.0, marker='o', label="delta_f_external [mHz]", linewidth=1.5)
        plt.xlabel("Time (s)")
        plt.ylabel("Delta f external (mHz)")
        plt.title("Delta f External vs Time (Slow Tracker)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


    else:
        raise ValueError(f"Unknown method: {method}. Use 'adaptive_tracking', 'enhanced_notch', or 'hybrid'")
    
    # Step 5: Analyze results
    if plot_results:
        suppression_db = analyze_interference_removal(
            data, cleaned_data, subtracted_signal, fs, f_interference
        )
        print(f"Achieved {suppression_db:.1f} dB suppression using {method} method")
    
    # Step 6: Save if requested
    import warnings
    if output_file is not None:
        save_cleaned_miniseed(data, stats, actual_start, output_file+"0")
        save_cleaned_miniseed(cleaned_data, stats, actual_start, output_file)
    
    return cleaned_data, data, stats

# ============================================================
if __name__ == "__main__":
    import os
    
    dir = r"/home/jbeale/Documents/rshake"
    outdir = r"/home/jbeale/Documents/rshake"
    fname = "AM.R2543.00.EHN.D.2025.161"
    #fname = "AM.R2543.00.EHZ.D.2025.161"
    filename = os.path.join(dir, fname)
    fInt = None # 0.95, 1.00 or None to Auto-detect interference frequency
    tStart = "2025-06-10T05:42:00"
    tEnd = "2025-06-10T05:56:00"
    print(f"Processing file: {filename}")
    print(f"Time range: {tStart} to {tEnd}")

    outPath = os.path.join(outdir, fname+"_cleaned.mseed")

    try:
        cleaned1, original1, stats1 = process_miniseed_file_adaptive(
            filename,
            start_time=tStart,
            end_time=tEnd,
            trace_number=4,
            f_interference=fInt,
            output_file=outPath,
            plot_results=True,            
            method='simple_continuous_tracking'
        )
        
    except FileNotFoundError:
        print("Input file not found. Please check the path and filename.")
