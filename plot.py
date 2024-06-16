import torchaudio
import plotly.graph_objects as go

# Load the audio file
audio_file = r"C:\Users\denou\Downloads\UrbanSound8K\UrbanSound8K\audio\fold4\74723-3-0-0.wav"
waveform, sample_rate = torchaudio.load(audio_file, normalize=True)

# Generate the spectrogram
transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=1024,
    hop_length=None,
    n_mels=64
)
spectrogram = transform(waveform)
spectrogram = torchaudio.transforms.AmplitudeToDB()(spectrogram)

# Combine the channels (e.g., by averaging)
combined_spectrogram = spectrogram.mean(dim=0).numpy()

def plot_spectrogram(spectrogram, title="Spectrogram"):
    fig = go.Figure(data=go.Heatmap(
        z=spectrogram,
        colorscale='inferno'
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Time (frames)',
        yaxis_title='Frequency (Hz)',
    )
    fig.show()

# Plot the combined spectrogram
plot_spectrogram(combined_spectrogram, title="Combined Spectrogram")
