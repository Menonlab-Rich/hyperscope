import numpy as np
from metavision_core.event_io import RawReader
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.sparse import coo_matrix, csr_matrix

def new_volume(events, tb, height, width):
    max_time = events['t'].max()
    num_bins = max_time // tb + 1
    volume = csr_matrix((num_bins, 2 * height * width), dtype=np.float32)
    return volume

def manual_histo_sparse(events, volume, tb, height, width):
    t_bins = events['t'] // tb
    y = events['y']
    x = events['x']
    p = events['p']

    for polarity in [0, 1]:
        mask = (p == polarity)
        row_indices = t_bins[mask]
        col_indices = y[mask] * width + x[mask] + polarity * height * width
        data = np.ones(len(row_indices), dtype=np.float32)

        volume += coo_matrix((data, (row_indices, col_indices)), shape=volume.shape).tocsr()

def snr(file):
    rates = []
    dt = int(1e6)  # 1s
    record = RawReader(file)
    events = record.load_delta_t(dt)
    events['t'] -= events[0]['t']  # Important! Almost all preprocessing use relative time!
    height, width = record.get_size()

    print(f"Loaded events. Number of events: {len(events)}, height: {height}, width: {width}")

    tbins = np.arange(100, 1100, 100)  # Time bins from 100us to 1000us

    for tb in tqdm(tbins):
        print(f"Processing time bin: {tb} us")
        volume = new_volume(events, tb, height, width)
        print(f"Volume shape before manual histo: {volume.shape}")

        manual_histo_sparse(events, volume, tb, height, width)
        print(f"Volume shape after manual histo: {volume.shape}")

        # Combine the two polarities and calculate the event rate
        combined_volume = volume[:, :height*width] + volume[:, height*width:]
        mean_volume = combined_volume.mean(axis=0).A.flatten()
        event_rate = np.count_nonzero(mean_volume) / mean_volume.size
        rates.append(event_rate)
    return rates

def plot_snr(rates, n_rates):
    snr = np.array(rates) / np.array(n_rates)
    plt.plot(np.arange(100, 1100, 100), snr)
    plt.xlabel('Time bin (us)')
    plt.ylabel('SNR')
    plt.title('Signal to Noise Ratio')
    plt.savefig('snr.png')

if __name__ == '__main__':
    signal = r'D:\CZI_scope\code\data\raw\605_capillary_flow_cut_2024-06-07.raw'
    noise = r'D:\CZI_scope\code\data\raw\recording_2024-05-06_07-36-55.raw'
    rate = snr(signal)
    noise_rate = snr(noise)
    plot_snr(rate, noise_rate)
