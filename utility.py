import dateparser
import re

def validate_input(value, value_type):
    if value_type == "email":
        if re.match(r"[^@]+@[^@]+\.[^@]+", value):
            return True
        else:
            return False
    elif value_type == "phone":
        if re.match(r"^\+?\d{10,15}$", value):
            return True
        else:
            return False
    return False


def extract_date_from_query(query):
    print(query)
    date = dateparser.parse(query,settings={'TIMEZONE': 'Asia/Kathmandu', 'RETURN_AS_TIMEZONE_AWARE': True})
    print(date)
    if date:
        return date.strftime("%Y-%m-%d")
    else:
        return None