import requests
import numpy as np

url = 'http://localhost:9696/predict'

seed_id = '2211'
seed = {    
    'area': 95971,
    'perimeter':1303.774,
    'major_axis_length': 558.417,
    'minor_axis_length': 219.8414,
    'convex_area': 96950,
    'equiv_diameter': 349.5627,
    'eccentricity': 0.9192,
    'solidity': 0.9899,
    'extent': 0.7065,
    'roundness': 0.7095,
    'aspect_ration': 2.5401,
    'compactness': 0.626
}

response = requests.post(url, json=seed).json()
values = list(response.values())
array = np.array(values)
    
    
if array[0] == 1.0:
    print('The seed type for sample %s is: Ürgüp Sivrisi' % seed_id)
elif array[0] == 0.0:
    print('The seed type for sample %s is: Çerçevelik' % seed_id)
else:
    print('Error')