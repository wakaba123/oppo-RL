import subprocess
import time

def execute(cmd):
    output = subprocess.checkout_output('hdc shell ' + f"\"{cmd}\"", shell=True).decode('utf-8').strip()
    return output

def get_fps():
    cmd = "hidumper -s RenderService -a 'composer fps'"
    output = execute(cmd)
    timestamps = output.split('\n')
    last_timestamp = timestamps[-1]
    global pre_timestamp
    if(pre_timestamp == last_timestamp):
        return -1
    pre_timestamp = last_timestamp
    count = 0
    for timestamp in timestamps:
        if int(timestamp) - int(last_timestamp) > 1e9:
            break
        count += 1
    return count 