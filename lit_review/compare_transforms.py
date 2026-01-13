import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

#configuration
AUDIO_PATH = 'tracks/Let It Be.mp3'
SR = 22050
HOP_LENGTH = 512

def main():
    if not os.path.exists(AUDIO_PATH):
        print(f"error: {AUDIO_PATH} not found.")
        return

    print(f"loading {AUDIO_PATH}...")
    y, sr = librosa.load(AUDIO_PATH, sr=SR)
    
    #only analyse the first 10-15 seconds for better visualisation
    y = y[:sr*15]

    #compute STFT (linear frequency)
    print("computing STFT...")
    D = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP_LENGTH))
    S_db = librosa.amplitude_to_db(D, ref=np.max)

    #compute CQT (logarithmic frequency)
    print("computing CQT...")
    C = np.abs(librosa.cqt(y, sr=sr, hop_length=HOP_LENGTH))
    C_db = librosa.amplitude_to_db(C, ref=np.max)

    #visualisation
    plt.figure(figsize=(12, 10))

    #plot STFT
    plt.subplot(2, 1, 1)
    librosa.display.specshow(S_db, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='linear')
    plt.title('Short-Time Fourier Transform (STFT) - Linear Frequency')
    plt.colorbar(format='%+2.0f dB')
    plt.ylim(0, 8000) #focus on musical frequency range

    #plot CQT
    plt.subplot(2, 1, 2)
    librosa.display.specshow(C_db, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='cqt_note')
    plt.title('Constant-Q Transform (CQT) - Logarithmic (Musical) Frequency')
    plt.colorbar(format='%+2.0f dB')

    plt.tight_layout()
    plt.savefig('stft_vs_cqt.png')
    print("\ncomparison graph saved to stft_vs_cqt.png")
    #plt.show() #skip show to avoid blocking

if __name__ == "__main__":
    main()
