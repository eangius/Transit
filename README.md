# Transit Schedules
<img src="images/scott-webb-Ms6bSWTcVkg-unsplash.jpg" width="100%">

Using historical data from the City Of Winnipeg, this repository performs exploratory analysis to
identify inefficiencies in the city's transit system with regards to scheduled times. We leverage
these findings to then design a real-time bus tardiness prediction system for individual bus stops.

## Getting Started
```sh
pip install -r requirements.txt;       # download per project external libs
env PYTHONPATH=${PWD} jupyter lab;     # launch interactive notebook environment
```


## Resources
* 🔢 [Transit Delays](https://data.winnipeg.ca/Transit/Transit-On-Time-Performance-Data-Archive/cymk-nyei/data_preview) - Primary data from City of Winnipeg
* 🔢 [Road Network](https://open.canada.ca/data/en/dataset/3d282116-e556-400c-9306-ca1a3cada77f/resource/7e43999a-a432-46df-83f1-f5ddb88ccdc7) - Auxiliary data from Government of Canada
* 🔢 [POIs](http://download.geofabrik.de/north-america/canada/manitoba.html) - Auxiliary data from Open Street Map
* 🛠️ [QGIS Editor](https://www.qgis.org/en/site/) - Mapping tool for exploration & data processing
