import re

response = "On server 10.20.64.181, the spike in CPU and memory utilization was caused by the file \"WfpMonitor.sys\", which is an IBM S-TAP application file. "
ip_addr_regex = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')
print (ip_addr_regex)
match = re.sub(ip_addr_regex, "a.b.c.d", response)
print ("hello")