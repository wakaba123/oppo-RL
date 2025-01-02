import re

def extract_metrics(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Extract total frames rendered
    total_frames_rendered = int(re.search(r'Total frames rendered: (\d+)', content).group(1))
    
    # Extract janky frames
    janky_frames = int(re.search(r'Janky frames: (\d+)', content).group(1))
    
    # Extract uptime
    uptime = int(re.search(r'Uptime: (\d+)', content).group(1))
    
    # Calculate jank rate
    jank_rate = (janky_frames / total_frames_rendered) * 100
    
    # Calculate frame rate
    frame_rate = (total_frames_rendered / uptime) * 1000
    
    return jank_rate, frame_rate

file_path = 'c:/Users/wakaba/Desktop/oppo-RL/temp.txt'
jank_rate, frame_rate = extract_metrics(file_path)
print(f"Jank Rate: {jank_rate:.2f}%")
print(f"Frame Rate: {frame_rate:.2f} fps")