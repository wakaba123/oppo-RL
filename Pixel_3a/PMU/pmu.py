from threading import Thread
import subprocess
import time
import re

def execute(cmd):
    cmds = [ 'su',cmd, 'exit']
    obj = subprocess.Popen("adb shell", shell= True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    info = obj.communicate(("\n".join(cmds) + "\n").encode('utf-8'))
    return info[0].decode('utf-8')

class PMUGet():
    def __init__(self, pid_list):
        self.instructions = []
        self.cycles = []
        self.cache_refs = []
        self.cache_misses = []    
        self.page_faults= []    
        self.pid_list = pid_list
        self.result = []

    def start(self):
        pmu_thread = Thread(target=self.get_pmu, args=())
        pmu_thread.start()

    def get_pmu(self):
        while True:
            duration = 0.5
            results = execute(f'simpleperf stat --group task-clock,instructions --group cache-references,cache-misses -e page-faults -t {self.pid_list} --duration {duration}')
            pattern = re.compile(r'\s*([\d\.,]+(?:\(\w+\))?)\s+([a-zA-Z\-]+)\s+#\s+([\d\.]+)')

            # 匹配并存储结果
            metrics = {}
            for match in pattern.finditer(results):
                count = match.group(1).replace(',', '').replace('(ms)', '').strip()
                event_name = match.group(2)
                rate = match.group(3)
                metrics[event_name] = {'count': float(count), 'rate': float(rate)}

            # 打印结果
            result = []
            for event, data in metrics.items():
                # print(f"{event}: Count = {data['count']}, Rate = {data['rate']}")
                result.append(data['count'])

            # [page-faults, task-clock, instructions, cache-references, cache-misses]
            self.result = result

if __name__ == '__main__':
    pmu = PMUGet('27665')
    pmu.start()
    time.sleep(2)
    print(pmu.result)
