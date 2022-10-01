DATA="""lat,long,city,country,year
49.279688143314516, -123.11540577269238,Vancouver,Canada,2022
21.37486438118298, -157.86796191057593,Hawaii,2019
37.38586811294064, -122.08041938589838,San Francisco,United States,2017
-6.203366850819404, 106.84250166232667,Jakarta,Indonesia,2019
17.067479135780257, 96.19431036542245,Yangon,Myanmar,2018
1.1534532193777867, 104.45258867666682,Bintan,Indonesia,2017
10.822647285594394, 106.60819911963733,Ho Chi Minh,Vietnam,2016
3.138605047730204, 101.69089826290852,Kuala Lumpur,Malaysia,2016
2.1953635883882665, 102.25056671502006,Malacca,Malaysia,2016
16.05391373170721, 108.20483204935474,Da Nang,Vietnam,2015
35.04517363790297, 135.7699580329708,Kyoto,Japan,2015
1.3483191278971893, 103.8566036956293,Singapore,Singapore,2015
11.970337375897662, 108.47074150647333,Dalat,Vietnam,2012
12.253439191393719, 109.19296838311456,Nha Trang,Vietnam,2012
21.027531976743198, 105.83523097920317,Hanoi,Vietnam,2003
16.467706196268217, 107.59098165211023,Hue,Vietnam,1999
17.744068951948165, 106.43337391758689,Quang Binh,Vietnam,1994
"""


import io
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from adjustText import adjust_text


df = pd.read_csv(io.StringIO(DATA), sep=",")
df["geometry"] = df.apply(lambda x: Point(float(x.long), float(x.lat)), axis=1)
df = gpd.GeoDataFrame(df)
print(df)

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

fig, ax = plt.subplots(figsize=(24,13))
world.plot(ax=ax, alpha=0.4, color='grey')
df.plot(ax=ax, color="orange")
texts = []
for t in df.itertuples(index=False):
    texts.append(plt.text(s=t[2], x=t[1], y=t[0] + 1, horizontalalignment='center', fontdict={'weight': 'normal', 'size': 8}))
adjust_text(texts)
plt.axis('off')
plt.tight_layout()
fig.savefig("assets/traveled_world.png")