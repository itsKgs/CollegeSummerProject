import mne
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#Load the data
raw = mne.io.read_raw_edf(r"C:\Users\kunwa\Downloads\chb01_01.edf", preload=True)

# Show basic info
print(raw.info)
raw.plot(
    duration=10,           # Duration to show per screen
    n_channels=10,         # Number of channels shown
    scalings='auto',       # Auto scale signal amplitude
    title='EEG Plot',      # Title for the window
    show=True              # Open the plot window
)

data_orig, times = raw.get_data(return_times=True)

raw.filter(1., 40.)  # Bandpass
raw.filter(1., 40.).plot(
    duration=10,           # Duration to show per screen
    n_channels=10,         # Number of channels shown
    scalings='auto',       # Auto scale signal amplitude
    title='EEG Plot',      # Title for the window
    show=True 
)

# Get data
data, times = raw.get_data(return_times=True)

# Visualize
raw.plot(scalings='auto')
plt.show()

import pandas as pd

# Get data and times
data, times = raw.get_data(return_times=True)
ch_names = raw.ch_names

# Create a DataFrame
df = pd.DataFrame(data.T, columns=ch_names)
df["time"] = times


# Save to CSV
df.to_csv(r"C:\Users\kunwa\Documents\Programming\Project\Brain\chb01_01_filtered.csv", index=False)
print("Saved EEG data to CSV!")

# Plotting 1 channel (e.g., first channel) for first 5 seconds
channel_idx = 0  # First EEG channel
fs = int(raw.info['sfreq'])  # Sampling rate
print(fs)
start = 0
stop = 5 * fs

plt.figure(figsize=(12, 5))
plt.plot(times[start:stop], data_orig[channel_idx, start:stop], label="Unfiltered", alpha=0.6)
plt.plot(times[start:stop], data[channel_idx, start:stop], label="Filtered (1–40 Hz)", linewidth=2)
plt.title(f"Comparison of EEG Signals - Channel: {raw.ch_names[channel_idx]}")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (µV)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()