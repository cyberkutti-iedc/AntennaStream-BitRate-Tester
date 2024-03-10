import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QLineEdit, QPushButton, QFileDialog
from PyQt5.QtGui import QIcon  # Add this import statement

class GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('AntennaStream BitRate Tester')
        self.setGeometry(100, 100, 500, 500)

        # Set application icon
        self.setWindowIcon(QIcon('icon.png'))

        layout = QVBoxLayout()

        # Labels and input fields
        labels = ['Distance (m):', 'Frequency (GHz):', 'Bandwidth (GHz):',
                  'Transmitted Power (W):', 'Text File:', 'CSV File:', 'Binary Data (10-64 bits):']
        self.text_fields = []

        for label_text in labels:
            label = QLabel(label_text)
            text_field = QLineEdit()
            layout.addWidget(label)
            layout.addWidget(text_field)
            self.text_fields.append(text_field)

        # Buttons for selecting text and CSV files
        text_file_button = QPushButton('Browse Text File')
        text_file_button.clicked.connect(self.browseTextFile)
        layout.addWidget(text_file_button)

        csv_file_button = QPushButton('Browse CSV File')
        csv_file_button.clicked.connect(self.browseCSVFile)
        layout.addWidget(csv_file_button)

        # Submit button
        submit_button = QPushButton('Submit')
        submit_button.clicked.connect(self.submitClicked)
        layout.addWidget(submit_button)

        self.setLayout(layout)

    def browseTextFile(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Text File', '', 'Text Files (*.txt)')
        if file_path:
            self.text_fields[4].setText(file_path)

    def browseCSVFile(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open CSV File', '', 'CSV Files (*.csv)')
        if file_path:
            self.text_fields[5].setText(file_path)

    def submitClicked(self):
        # Gather input values
        inputs = {}
        for label, text_field in zip(['distance', 'frequency', 'bandwidth',
                                      'transmitted_power', 'txtfile', 'csvfile', 'binary_data'], self.text_fields):
            inputs[label] = text_field.text()

        # Process the inputs and display results
        processInputs(inputs)


def generate_bpsk_signal_with_data(data, symbol_rate, duration, amplitude, frequency):
    t = np.linspace(0, duration, int(symbol_rate * duration), endpoint=False)
    symbols = 2 * data - 1
    symbols_repeated = np.tile(symbols, len(t) // len(symbols))

    if len(symbols_repeated) != len(t):
        symbols_repeated = np.append(symbols_repeated, symbols[:len(t) - len(symbols_repeated)])

    modulated_signal = amplitude * np.cos(2 * np.pi * frequency * t) * symbols_repeated

    return t, modulated_signal


def bpsk_modulation(binary_data):
    return 2 * binary_data - 1


def custom_bpsk_demodulation(received_signal):
    return np.where(received_signal < 0, 0, 1)


def calculate_data_rate(bandwidth, signal_power, noise_power, gain_total_from_csv, lam, L, distance):
    snr = (signal_power * gain_total_from_csv * gain_total_from_csv * lam**2) / ((4 * np.pi)**2 * L * distance**2)
    data_rate_bps = bandwidth * np.log2(1 + snr)
    data_rate_gbps = data_rate_bps / 1e9  # Convert bps to Gbps
    return data_rate_bps, data_rate_gbps


def processInputs(inputs):
    distance = float(inputs['distance'])
    frequency = float(inputs['frequency']) * 1e9  # Convert GHz to Hz
    bandwidth = float(inputs['bandwidth']) * 1e9  # Convert GHz to Hz
    transmitted_power = float(inputs['transmitted_power'])
    txtfile = inputs['txtfile']
    csvfile = inputs['csvfile']
    binary_data_str = inputs['binary_data']

    # Validate binary data input
    if not all(bit.isdigit() and int(bit) in [0, 1] for bit in binary_data_str):
        print("Invalid binary data input. Please enter a sequence of 0s and 1s.")
        return
    if len(binary_data_str) < 10 or len(binary_data_str) > 64:
        print("Binary data length should be between 10 and 64 bits.")
        return

    binary_data = np.array([int(bit) for bit in binary_data_str])

    # Other parameters
    noise_power = 1e-9  # Noise power in watts
    symbol_rate = 5e9  # 5 Gbps symbol rate
    duration = 16 / symbol_rate
    amplitude = 1.0
    modulation_order = 2  # BPSK modulation
    output_json = 'output.json'
    modulation_output_json = 'modulation_output.json'
    L = 1
    lam = int(3 * 10 ** 8) / frequency

    df = pd.read_csv(csvfile, names=["Theta [deg]", "dB(GainTotal) [] - Freq='60GHz' Phi='90deg'"], skiprows=1)
    max_gain_row = df.loc[df["dB(GainTotal) [] - Freq='60GHz' Phi='90deg'"].idxmax()]
    theta_from_csv = max_gain_row["Theta [deg]"]
    gaintotal_from_csv = max_gain_row["dB(GainTotal) [] - Freq='60GHz' Phi='90deg'"]

    hfss_parameters = {}
    with open(txtfile, 'r') as file:
        for line in file:
            words = line.split()
            if "dB(GainTotal)" in line:
                hfss_parameters["Antenna Gain"] = float(words[-1])
            elif "Max U" in line:
                hfss_parameters["Max U"] = float(words[-2])
            elif "Peak Directivity" in line:
                hfss_parameters["Peak Directivity"] = float(words[-1])
            elif "Peak Gain" in line:
                hfss_parameters["Peak Gain"] = float(words[-1])
            elif "Peak Realized Gain" in line:
                hfss_parameters["Peak Realized Gain"] = float(words[-1])
            elif "Radiated Power" in line:
                hfss_parameters["Radiated Power"] = float(words[-2])
            elif "Accepted Power" in line:
                hfss_parameters["Accepted Power"] = float(words[-2])
            elif "Incident Power" in line:
                hfss_parameters["Incident Power"] = float(words[-2])
            elif "Radiation Efficiency" in line:
                hfss_parameters["Radiation Efficiency"] = float(words[-1])
            elif "Front to Back Ratio" in line:
                hfss_parameters["Front to Back Ratio"] = float(words[-1])
            elif "Decay Factor" in line:
                hfss_parameters["Decay Factor"] = float(words[-1])

    t, modulated_signal = generate_bpsk_signal_with_data(binary_data, symbol_rate, duration, amplitude, frequency)
    bit_rate = symbol_rate / modulation_order

    if distance > 0.5:
        print("Signal strength is poor for the receiver to demodulate.")
        return

    received_signals = modulated_signal + 0.1 * np.random.normal(size=len(modulated_signal))
    demodulated_output = custom_bpsk_demodulation(received_signals)
    all_demodulated_outputs = [custom_bpsk_demodulation(received_signal) for received_signal in received_signals]

    all_parameters = {
        "theta_from_csv": theta_from_csv,
        "gaintotal_from_csv": gaintotal_from_csv,
        **hfss_parameters,
        "frequency": frequency,
        "binary_data": binary_data.tolist(),
        "modulated_signal": modulated_signal.tolist(),
        "received_signals": received_signals.tolist(),
        "all_demodulated_outputs": [output.tolist() for output in all_demodulated_outputs]
    }

    signalpower = hfss_parameters["Accepted Power"]
    data_rate_bps, data_rate_gbps = calculate_data_rate(bandwidth, signalpower, noise_power, gaintotal_from_csv, lam,
                                                        L, distance)

    print("Parameters extracted:")
    for key, value in all_parameters.items():
        print(f"{key}: {value}")

    with open(output_json, 'w') as json_file:
        json.dump(all_parameters, json_file, indent=4)

    modulation_output = {
        "binary_data": binary_data.tolist(),
        "modulated_signal": modulated_signal.tolist(),
        "received_signals": received_signals.tolist(),
        "all_demodulated_outputs": [output.tolist() for output in all_demodulated_outputs]
    }

    with open(modulation_output_json, 'w') as modulation_json_file:
        json.dump(modulation_output, modulation_json_file, indent=4)

    print("Using the Shannon-Hartley theorem, the data rate is:", data_rate_bps, "bps")
    print("Using the Shannon-Hartley theorem, the data rate is:", data_rate_gbps, "Gbps")

    # Calculate data rates for each time point
    data_rates_per_time = []
    for i in range(len(received_signals)):
        data_rate_gbps, _ = calculate_data_rate(bandwidth, signalpower, noise_power, gaintotal_from_csv, lam, L, distance)
        data_rates_per_time.append(data_rate_gbps)

    

        # Plotting Input Binary Data
    plt.figure(figsize=(8, 6))
    plt.stem(np.arange(len(binary_data)), binary_data)
    plt.title('Input Binary Data')
    plt.xlabel('Data Index')
    plt.ylabel('Binary Value')
    plt.tight_layout()
    plt.show()

    # Plotting Modulated Signal
    plt.figure(figsize=(8, 6))
    plt.plot(t, modulated_signal)
    plt.title('Modulated Signal')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()

    # Plotting Received Signals
    plt.figure(figsize=(8, 6))
    plt.plot(t, received_signals)
    plt.title('Received Signals')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()

    # Plotting Demodulated Output
    plt.figure(figsize=(8, 6))
    plt.stem(t, all_demodulated_outputs)
    plt.title('Demodulated Output')
    plt.xlabel('Time (ms)')
    plt.ylabel('Binary Value')
    plt.tight_layout()
    plt.show()

    # Plotting Data Rate vs Time
    plt.figure(figsize=(8, 6))
    plt.plot(t, data_rates_per_time, color='orange', marker='o')
    plt.title('Data Rate vs Time')
    plt.xlabel('Time (ms)')
    plt.ylabel('Data Rate (Gbps)')
    plt.annotate(f'Distance: {distance} m', xy=(1, 1.1), xycoords='axes fraction', ha='center', fontsize=12, color='blue')
    plt.tight_layout()
    plt.show()


   
    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = GUI()
    gui.show()
    sys.exit(app.exec_())
