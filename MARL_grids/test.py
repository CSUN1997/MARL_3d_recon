from subprocess import check_output
import re

def get_ip_addresses():
    #this function gets WIFI IP addresses of the computer
    string = str(check_output(['ipconfig']))
    reg = r'Wireless LAN adapter Wi-Fi.+?(?=Subnet Mask)'
    search_results = re.findall(reg, string)
    add_dict = {}
    for res in search_results:
        start0, end0 = re.search('.+?(?=:)', res.strip().split('\\n')[0]).span()
        start1, end1 = re.search('([0-9]+\.){3}[0-9]+',res.strip().split('\\n')[-2]).span()
        add_dict[res.strip().split('\n')[0][start0:end0]] = res.strip().split('\\n')[-2][start1:end1]

    return add_dict

