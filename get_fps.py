import os
import re
import time

count = 0

def get_gfxinfo():
    global count

    result = os.popen("adb shell dumpsys gfxinfo com.ss.android.ugc.aweme framestats").read()
    with open(f"temp_{count}.txt","w") as f:
        f.write(result)
    count += 1
    return result

def parse_gfxinfo(gfxinfo):
    uptime = int(re.search(r'Uptime: (\d+)', gfxinfo).group(1))
    # activity_info = list(re.finditer(r'Window: com\.ss\.android\.ugc\.aweme/com\.ss\.android\.ugc\.aweme\.detail\.ui\.DetailActivity', gfxinfo))
    # if activity_info:
    #     # Extract the relevant section after the last activity_info match
    #     print(len(activity_info))
    #     relevant_section = gfxinfo[activity_info[-1].end():]
    #     total_frames_rendered = int(re.search(r'Total frames rendered: (\d+)', relevant_section).group(1))
    #     janky_frames = int(re.search(r'Janky frames: (\d+)', relevant_section).group(1))
    #     return uptime, total_frames_rendered, janky_frames
        # Extract the relevant section after the last activity_info match
    relevant_section = gfxinfo
    total_frames_rendered = int(re.search(r'Total frames rendered: (\d+)', relevant_section).group(1))
    janky_frames = int(re.search(r'Janky frames: (\d+)', relevant_section).group(1))
    return uptime, total_frames_rendered, janky_frames


def main():
    interval = 1 # 60 seconds interval
    previous_uptime = None
    previous_total_frames = None
    previous_janky_frames = None

    while True:
        gfxinfo = get_gfxinfo()
        uptime, total_frames_rendered, janky_frames = parse_gfxinfo(gfxinfo)
        print(uptime, total_frames_rendered, janky_frames)

        if previous_uptime is not None and uptime is not None:
            uptime_diff = uptime - previous_uptime
            frames_diff = total_frames_rendered - previous_total_frames
            janky_diff = janky_frames - previous_janky_frames

            avg_frames = frames_diff / (uptime_diff / 1000)
            avg_janky = janky_diff / (uptime_diff / 1000)
            print(f"Count: {count}")
            print(f"Average Frames: {avg_frames:.2f} fps")
            print(f"Average Janky Frames: {avg_janky:.2f} fps")

        previous_uptime = uptime
        previous_total_frames = total_frames_rendered
        previous_janky_frames = janky_frames

        time.sleep(interval)

if __name__ == "__main__":
    main()