import sys
import struct
import uselect
from machine import Pin, SPI, I2S, PWM, freq
import time

# Set CPU to exactly 144 MHz
freq(144_000_000) 

class ADF4351:
    def __init__(self, spi, cs_pin, ref_clk=32.0):
        self.spi = spi
        self.cs = Pin(cs_pin, Pin.OUT)
        self.cs.value(1)
        self.ref_clk = ref_clk
        
    def write_reg(self, reg):
        buf = bytearray(4)
        buf[0], buf[1], buf[2], buf[3] = (reg >> 24) & 0xFF, (reg >> 16) & 0xFF, (reg >> 8) & 0xFF, reg & 0xFF
        self.cs.value(0)
        self.spi.write(buf)
        self.cs.value(1)
        
    def init_registers(self):
        self.write_reg(0x00580005) 
        self.write_reg(0x0000000B) 
        self.write_reg(0x18004E42) 

    def set_frequency(self, freq_mhz, power_level=3):
        vco = float(freq_mhz)
        div_bits = 0
        while vco < 2200.0 and div_bits < 7:
            vco *= 2
            div_bits += 1
            
        MOD = 4000
        n_total = vco / self.ref_clk
        INT, FRAC = int(n_total), int(round((n_total - int(n_total)) * MOD))
        bscc = 255 
        
        self.write_reg((1 << 23) | (div_bits << 20) | (bscc << 12) | (1 << 5) | (power_level << 3) | 4)
        self.write_reg((1 << 15) | (MOD << 3) | 1)
        self.write_reg((INT << 15) | (FRAC << 3) | 0)

# Synthesizer Setup
spi_lo = SPI(0, baudrate=1_000_000, polarity=0, phase=0, sck=Pin(2), mosi=Pin(3))
synth_lo = ADF4351(spi_lo, cs_pin=1)
spi_rf = SPI(1, baudrate=1_000_000, polarity=0, phase=0, sck=Pin(10), mosi=Pin(11))
synth_rf = ADF4351(spi_rf, cs_pin=9)

synth_lo.init_registers()
synth_rf.init_registers()

# Hardware Lock Detect Pins
lock_lo = Pin(4, Pin.IN, Pin.PULL_DOWN)
lock_rf = Pin(12, Pin.IN, Pin.PULL_DOWN)

# Master Clock Generation
scki = PWM(Pin(21))
scki.freq(24000000) 
scki.duty_u16(32768) 

# I2S DMA Setup
audio = I2S(0, sck=Pin(19), ws=Pin(20), sd=Pin(22), 
            mode=I2S.RX, bits=32, format=I2S.STEREO, rate=93750, ibuf=8192)
i2s_buf = bytearray(2048) 

poller = uselect.poll()
poller.register(sys.stdin, uselect.POLLIN)
IF_OFFSET_MHZ = 0.024 

try:
    while True:
        if poller.poll(0):
            cmd = sys.stdin.buffer.read(1)
            
            if cmd == b'F': 
                freq_bytes = bytearray()
                
                while len(freq_bytes) < 4:
                    chunk = sys.stdin.buffer.read(4 - len(freq_bytes))
                    if chunk:
                        freq_bytes.extend(chunk)
                    else:
                        time.sleep(0.001)
                
                if len(freq_bytes) == 4:
                    freq_lo = struct.unpack('<f', freq_bytes)[0]
                    freq_rf = freq_lo + IF_OFFSET_MHZ
                    
                    synth_lo.set_frequency(freq_lo, power_level=3)
                    synth_rf.set_frequency(freq_rf, power_level=2)
                    
                    # 1. Active Digital Lock Polling
                    # Actively drain the buffer so it does not fill up while waiting for the silicon to assert the lock pins.
                    t_start = time.ticks_ms()
                    while lock_lo.value() == 0 or lock_rf.value() == 0:
                        audio.readinto(i2s_buf)
                        if time.ticks_diff(time.ticks_ms(), t_start) > 50:
                            break
                    
                    # 2. Active Analog Settling Delay
                    # Each readinto() blocks for exactly 2.73ms. Looping 10 times yields a perfect 27.3ms delay.
                    # The I2S hardware buffer remains completely empty the entire time.
                    for _ in range(10):
                        audio.readinto(i2s_buf) 
                    
                    # 3. Final Capture
                    # Read the pristine, aligned, steady-state sine wave.
                    audio.readinto(i2s_buf) 
                    
                    sys.stdout.buffer.write(b'VNA1')
                    sys.stdout.buffer.write(struct.pack('<f', freq_lo))
                    sys.stdout.buffer.write(i2s_buf)

except KeyboardInterrupt:
    audio.deinit()
