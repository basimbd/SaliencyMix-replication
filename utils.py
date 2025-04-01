from datetime import datetime

def get_current_timestamp_string():
    return datetime.now().strftime('%Y%m%d')

if __name__ == '__main__':
    print(get_current_timestamp_string())
