import requests
url = 'http://127.0.0.1:8887/score'


r = requests.post(url,json=[5.605362, 21.510036, 'Tue', -2.687724, 100.359864, 'New York', 'ford'])
# r = requests.post(url,json=[ 21.510036, 'Tue', -2.687724, 100.359864, 'New York', 'ford'])
print(r.json())