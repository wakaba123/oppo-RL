import time
import Monsoon.HVPM as HVPM
import Monsoon.sampleEngine as sampleEngine
import Monsoon.Operations as op


# class PowerLogger:
# 	def __init__(self):
# 		self.power=0
# 		self.voltage=0
# 		self.current=0
# 		self.power_data = []
# 		self.voltage_data = []
# 		self.current_data = []
# 		self.Mon = HVPM.Monsoon()
# 		self.Mon.setup_usb()
# 		self.engine = sampleEngine.SampleEngine(self.Mon)
# 		self.engine.disableCSVOutput()
# 		self.engine.ConsoleOutput(False)

# 	def _getTime(self):
# 		return time.clock_gettime(time.CLOCK_REALTIME)

# 	def getPower(self):
# 		self.engine.enableChannel(sampleEngine.channels.MainCurrent)
# 		self.engine.enableChannel(sampleEngine.channels.MainVoltage)
# 		self.engine.startSampling(1)
# 		sample = self.engine.getSamples()
# 		current = sample[sampleEngine.channels.MainCurrent][0]
# 		voltage = sample[sampleEngine.channels.MainVoltage][0]
# 		self.Mon.stopSampling()
# 		self.engine.disableChannel(sampleEngine.channels.MainCurrent)
# 		self.engine.disableChannel(sampleEngine.channels.MainVoltage)
# 		self.power = current * voltage
# 		self.power_data.append(self.power)
# 		#print(self.power)
# 		return current * voltage

# 	def getVoltage(self):
# 		self.engine.startSampling(1)
# 		sample = self.engine.getSamples()
# 		voltage = sample[sampleEngine.channels.MainVoltage][0]
# 		self.Mon.stopSampling()
# 		self.voltage = voltage
# 		self.voltage_data.append(self.voltage)
# 		return voltage

# 	def getCurrent(self):
# 		self.engine.startSampling(1)
# 		sample = self.engine.getSamples()
# 		current = sample[sampleEngine.channels.MainCurrent][0]
# 		self.Mon.stopSampling()
# 		self.current = current
# 		self.current_data.append(self.current)
# 		return current

import subprocess

def execute(cmd):
    cmds = [ 'su',cmd, 'exit']
    obj = subprocess.Popen("adb shell", shell= True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    info = obj.communicate(("\n".join(cmds) + "\n").encode('utf-8'))
    return info[0].decode('utf-8')

class PowerLogger: # 上面的PowerLogger是原生的powerlogger, 但是我们目前不是所有设备都有powermonitor测试功耗，因此做一些移植
	# 准确性上会存在一定的问题，但是
	def __init__(self):
		self.power_data = []

	def getPower(self):
		# print(execute('cat /sys/class/power_supply/battery/voltage_now'))
		# print(execute('cat /sys/class/power_supply/battery/current_now'))
		voltage = float(execute('cat /sys/class/power_supply/battery/voltage_now')) / 1e6
		current = float(execute('cat /sys/class/power_supply/battery/current_now')) / 1e6
		print('voltage and current is {}, {}'.format(  voltage,current))
		self.power = current * voltage
		self.power_data.append(self.power)
		return current * voltage

if __name__ == '__main__':
	powerlogger = PowerLogger()
	print(powerlogger.getPower())