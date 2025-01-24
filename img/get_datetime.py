import re
import datetime
from pathlib import Path

def extract_datetime_from_filename(filename):
    dow_dict = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday"
    }
    
    filename = Path(filename).name
    pattern = r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})"
    match = re.search(pattern, filename)

    if match:
        year, month, day, hour, minute, second = match.groups()
        dt = datetime.datetime(int(year), int(month), int(day))
        dow = dow_dict[dt.weekday()]
        return f"{year}-{month}-{day} {hour}:{minute}:{second}", dow
    else:
        return "ファイル名が指定の形式に一致しません"

# 使用例
filename = "20240731_153711807_V.jpg"
time, dow = extract_datetime_from_filename(filename)

print(time, dow)