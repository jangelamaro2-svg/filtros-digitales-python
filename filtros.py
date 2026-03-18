import numpy as np
import matplotlib.pyplot as plt

# =========================
# Señal de prueba
# =========================
fs = 1000  # frecuencia de muestreo
t = np.linspace(0, 1, fs)

# Señal: mezcla de 50 Hz y 200 Hz
senal = np.sin(2*np.pi*50*t) + np.sin(2*np.pi*200*t)

# Ruido
ruido = np.random.normal(0, 0.5, t.shape)

# Señal con ruido
senal_ruido = senal + ruido

# =========================
# FFT (para simular filtros)
# =========================
fft = np.fft.fft(senal_ruido)
frecuencias = np.fft.fftfreq(len(fft), 1/fs)

# =========================
# Filtro pasa bajos
# =========================
fft_low = fft.copy()
fft_low[np.abs(frecuencias) > 100] = 0
senal_low = np.fft.ifft(fft_low)

# =========================
# Filtro pasa altos
# =========================
fft_high = fft.copy()
fft_high[np.abs(frecuencias) < 100] = 0
senal_high = np.fft.ifft(fft_high)

# =========================
# Filtro pasa bandas
# =========================
fft_band = fft.copy()
fft_band[(np.abs(frecuencias) < 80) | (np.abs(frecuencias) > 220)] = 0
senal_band = np.fft.ifft(fft_band)

# =========================
# Gráficas
# =========================

plt.figure()
plt.plot(t, senal_ruido)
plt.title("Señal con ruido")
plt.savefig("senal_ruido.png")

plt.figure()
plt.plot(t, senal_low.real)
plt.title("Filtro pasa bajos")
plt.savefig("pasa_bajos.png")

plt.figure()
plt.plot(t, senal_high.real)
plt.title("Filtro pasa altos")
plt.savefig("pasa_altos.png")

plt.figure()
plt.plot(t, senal_band.real)
plt.title("Filtro pasa bandas")
plt.savefig("pasa_bandas.png")

plt.show()