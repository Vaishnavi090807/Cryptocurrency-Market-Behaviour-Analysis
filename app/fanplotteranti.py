import serial
import re
import threading
import collections
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation

# --- Configuration ---
COM_PORT = 'COM12'       # Update to your outgoing Bluetooth COM port
BAUD_RATE = 9600
MAX_DATA_POINTS = 50    # Number of historical points shown on the graph

# Thread-safe ring buffer for plot points
data_history = collections.deque(maxlen=MAX_DATA_POINTS)
ser = None

# Regular expression to extract numeric temperature from the incoming message:
# Expected string: "Temp: 28.4C | Fan: LOW | Mode: AUTO"
temp_regex = re.compile(r"Temp:\s*([0-9]+\.[0-9]+)C")

def serial_reader():
    """Background worker to continuously read serial data without freezing GUI."""
    global ser
    while ser and ser.is_open:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue
            
            # Print to Python console for logging
            print(f"[RX] {line}")
            
            # Parse temperature value
            match = temp_regex.search(line)
            if match:
                temp_val = float(match.group(1))
                data_history.append(temp_val)
                lbl_current_temp.config(text=f"{temp_val:.1f} °C")
            
            # Update incoming raw log in the UI
            lbl_raw_log.config(text=line)

        except Exception as e:
            print(f"Read error: {e}")
            break

def send_command(cmd_char):
    """Sends a single-byte command via Bluetooth."""
    global ser
    if ser and ser.is_open:
        try:
            ser.write(cmd_char.encode('utf-8'))
            print(f"[TX] Sent command: {cmd_char}")
        except Exception as e:
            print(f"Transmission failed: {e}")
    else:
        print("Serial port not connected.")

def send_custom_entry(event=None):
    """Sends arbitrary single character from the input field."""
    text = entry_cmd.get().strip()
    if text:
        send_command(text[0])
        entry_cmd.delete(0, tk.END)

# --- GUI Setup ---
root = tk.Tk()
root.title("STM32 Smart Fan Bluetooth Controller")
root.geometry("800x650")

# Top Metrics Frame
frame_metrics = ttk.LabelFrame(root, text="System Telemetry")
frame_metrics.pack(fill=tk.X, padx=10, pady=5)

ttk.Label(frame_metrics, text="Current Temperature:", font=("Arial", 11)).grid(row=0, column=0, padx=5, pady=5, sticky="w")
lbl_current_temp = ttk.Label(frame_metrics, text="--.- °C", font=("Arial", 14, "bold"), foreground="#007ACC")
lbl_current_temp.grid(row=0, column=1, padx=5, pady=5, sticky="w")

ttk.Label(frame_metrics, text="Raw Telemetry:", font=("Arial", 10)).grid(row=1, column=0, padx=5, pady=2, sticky="w")
lbl_raw_log = ttk.Label(frame_metrics, text="Waiting for connection...", font=("Consolas", 9), foreground="#555")
lbl_raw_log.grid(row=1, column=1, columnspan=3, padx=5, pady=2, sticky="w")

# Interactive Control Panel Frame
frame_ctrl = ttk.LabelFrame(root, text="Command Center")
frame_ctrl.pack(fill=tk.X, padx=10, pady=5)

btn_auto = ttk.Button(frame_ctrl, text="Auto Mode (A)", command=lambda: send_command('A'))
btn_auto.grid(row=0, column=0, padx=5, pady=5)

btn_low = ttk.Button(frame_ctrl, text="Fan Low (L)", command=lambda: send_command('L'))
btn_low.grid(row=0, column=1, padx=5, pady=5)

btn_norm = ttk.Button(frame_ctrl, text="Fan Normal (N)", command=lambda: send_command('N'))
btn_norm.grid(row=0, column=2, padx=5, pady=5)

btn_high = ttk.Button(frame_ctrl, text="Fan High (H)", command=lambda: send_command('H'))
btn_high.grid(row=0, column=3, padx=5, pady=5)

btn_stop = ttk.Button(frame_ctrl, text="Fan Stop (S)", command=lambda: send_command('S'))
btn_stop.grid(row=0, column=4, padx=5, pady=5)

ttk.Label(frame_ctrl, text="Custom Char:").grid(row=0, column=5, padx=(15, 2), pady=5)
entry_cmd = ttk.Entry(frame_ctrl, width=5)
entry_cmd.grid(row=0, column=6, padx=2, pady=5)
entry_cmd.bind("<Return>", send_custom_entry)
btn_send = ttk.Button(frame_ctrl, text="Send", width=6, command=send_custom_entry)
btn_send.grid(row=0, column=7, padx=2, pady=5)

# Matplotlib Live Graph
fig, ax = plt.subplots(figsize=(7, 3.8), dpi=100)
line_plot, = ax.plot([], [], color="#E63946", linewidth=2, marker='o', markersize=4)
ax.set_title("Live Temperature Profile (°C)", fontsize=11)
ax.set_ylabel("Temp (°C)")
ax.set_xlabel("Time (Samples)")
ax.grid(True, linestyle="--", alpha=0.6)
ax.set_ylim(20, 45)  # Set expected operating range for LM35

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

def update_plot(frame):
    """Animation callback to update the graph series."""
    if len(data_history) > 0:
        y_data = list(data_history)
        x_data = list(range(len(y_data)))
        line_plot.set_data(x_data, y_data)
        ax.set_xlim(0, max(MAX_DATA_POINTS, len(y_data)))
        
        # Adjust dynamic Y-axis bounds if temp exceeds typical range
        min_temp = min(y_data)
        max_temp = max(y_data)
        if min_temp < ax.get_ylim()[0] or max_temp > ax.get_ylim()[1]:
            ax.set_ylim(min(20.0, min_temp - 2), max(40.0, max_temp + 2))
            
    return line_plot,

# Connect serial port and start background reader
try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1.0)
    print(f"Connected to {COM_PORT} at {BAUD_RATE} baud.")
    reader_thread = threading.Thread(target=serial_reader, daemon=True)
    reader_thread.start()
except Exception as err:
    lbl_raw_log.config(text=f"Connection Failed: {err}", foreground="red")
    print(f"Failed to open {COM_PORT}: {err}")

# Setup animation timer and start UI mainloop
ani = animation.FuncAnimation(fig, update_plot, interval=500, blit=False)

def on_closing():
    global ser
    if ser and ser.is_open:
        ser.close()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()