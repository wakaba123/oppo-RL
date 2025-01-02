import os
import csv

def extract_distribution_data(file_path):
    with open(file_path, 'r', encoding='utf-16') as file:
        lines = file.readlines()
        mean_fps = float(lines[2].split()[1])
        action_counts = len(lines[12:]) - 2
    return mean_fps, action_counts

def extract_power_data(file_path):
    with open(file_path, 'r', encoding='utf-16') as file:
        lines = file.readlines()
        power = float(lines[2].strip())
    return power

def main(model_prefix):
    output_file = model_prefix + '_aggregated_data.csv'
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Model Name', 'episode','Mean FPS', 'Action Count', 'Power']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for file_name in os.listdir('.'):
            if file_name.startswith(model_prefix) and file_name.endswith('_distribution_output.txt'):
                model_name = file_name.replace('_distribution_output.txt', '')
                mean_fps, action_counts = extract_distribution_data(file_name)
                power_file = f"{model_name}_power_output.txt"
                episode = model_name.split('_')[-1]
                model_type ="_".join(file_name.replace('_distribution_output.txt', '').split('_')[:-1])
                if os.path.exists(power_file):
                    power = extract_power_data(power_file)
                    writer.writerow({
                        'Model Name': model_type,
                        'episode' : episode,
                        'Mean FPS': mean_fps,
                        'Action Count': action_counts,
                        'Power': power
                    })

if __name__ == "__main__":
    import sys
    model_prefix = sys.argv[1] if len(sys.argv) > 1 else ''
    main(model_prefix)
