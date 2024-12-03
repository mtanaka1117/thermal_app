import os
import re
from datetime import datetime, timedelta

def extract_time_from_filename(filename):
    match = re.match(r'(\d{8})_(\d{6})(\d{3})\_T\.(dat)', filename)
    if match:
        yyyymmdd, hhmmss, fff, _ = match.groups()
        return datetime.strptime(f'{hhmmss}{fff}', '%H%M%S%f')
    return None


def collect_files_near_intervals(directory, interval_ms=50):
    all_files = os.listdir(directory)
    valid_files = [f for f in all_files]

    file_times = [(f, extract_time_from_filename(f)) for f in valid_files if extract_time_from_filename(f)]
    # file_times = sorted(file_times, key=lambda x: x[1])

    selected_files = []
    if not file_times:
        return selected_files

    current_time = file_times[0][1]
    
    itr = 0
    # while current_time <= file_times[-1][1]:
    while itr < len(file_times):
        if itr+5 < len(file_times):
            closest_file = min(file_times[itr : itr+5], key=lambda x: abs((x[1] - current_time).total_seconds() * 1000))
        else:
            closest_file = min(file_times[itr:], key=lambda x: abs((x[1] - current_time).total_seconds() * 1000))
        
        if closest_file not in selected_files:
            selected_files.append(closest_file[0])
            selected_files.append(closest_file[0].replace('T.dat', 'V.jpg'))
            itr = file_times.index(closest_file)+1
        current_time += timedelta(milliseconds=interval_ms)
    
    return [f for f in selected_files]


def delete_unselected_files(directory, selected_files):
    all_files = set(os.listdir(directory))
    selected_filenames = {f for f in selected_files}
    
    files_to_delete = all_files - selected_filenames
    for filename in files_to_delete:
        os.remove(os.path.join(directory, filename))

directory = './data/0712/table/'
selected_files = collect_files_near_intervals(directory)

# for file in selected_files:
#     print(file)

delete_unselected_files(directory, selected_files)