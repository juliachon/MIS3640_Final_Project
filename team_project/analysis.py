import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score

"""reading data"""
data = pd.read_csv(r"team_project/listings.csv") 
# data = pd.read_csv(r"listings.csv") 

"""dataframe of all variables except categorical"""
df_total = pd.DataFrame(data, columns = ['neighbourhood', 'room_type', 'minimum_nights','number_of_reviews', 'availability_365'])
# print(df_total)

"""created dataframe of relevant variables"""
df = pd.DataFrame(data, columns = ['neighbourhood', 'room_type', 'price', 'minimum_nights','number_of_reviews', 'latitude', 'longitude'])
#print(df) #3254 rows and 7 columns

#cleaning/checking data
"""checking datatype"""
# print(df.dtypes)
"""checking for null values"""
# print(df.isnull().sum())
"""renaming columns to make more sense"""
df_new = df.rename(columns={'neighbourhood':'Neighborhood','room_type':'Room_Type','price':'Price','minimum_nights':'Minimum_Nights','number_of_reviews':'Number_of_Reviews', 'latitude':"Latitude", 'longitude':'Longitude'})
# print(df_new)

"""check for amount of unique values"""
# print(df_new.Neighborhood.unique())
# print(len(df_new.Neighborhood.unique())) #25 Neighborhoods
# print(df_new.Room_Type.unique())

"""check to see which neighborhoods have the most Airbnb listings. shows top 10 neighborhoods"""
top_neighborhoods = df_new.Neighborhood.value_counts().head(10)
# print(top_neighborhoods)
"""create table to show data (for map analysis later)"""
top_neighborhoods_df=pd.DataFrame(top_neighborhoods)
top_neighborhoods_df.reset_index(inplace=True)
top_neighborhoods_df.rename(columns={'index':'Neighborhood','Neighborhood':'Number of Listings'}, inplace=True)
# print(top_neighborhoods_df)

"""BAR GRAPH"""
neighborhood_bar=sns.barplot(x='Neighborhood', y='Number of Listings',data=top_neighborhoods_df, palette='Greens_d')
neighborhood_bar.set_title('Number of Listings by Neighborhood')
neighborhood_bar.set_xlabel('Neighborhood')
neighborhood_bar.set_ylabel('Number of Listings')
neighborhood_bar.set_xticklabels(neighborhood_bar.get_xticklabels(), rotation=45)
#plt.show()

"""price distribution by Neighborhood on a map"""
"""convert latitdue and longitude to mercator values to plot on a map"""
average_prices_df = pd.read_csv(r"team_project/average_prices_final.csv") 
# average_prices_df = pd.read_csv(r"average_prices_final.csv") 

"""Dorchester"""
dorchester_price=df_new.loc[df_new['Neighborhood'] == 'Dorchester']
price_dorchester=dorchester_price[['Price']]
"""Downtown"""
downtown_price=df_new.loc[df_new['Neighborhood'] == 'Downtown']
price_downtown=downtown_price[['Price']]
"""Jamaica Plain"""
jamaicaplain_price=df_new.loc[df_new['Neighborhood'] == 'Jamaica Plain']
price_jamaicaplain=jamaicaplain_price[['Price']]
"""Brighton"""
brighton_price=df_new.loc[df_new['Neighborhood'] == 'Brighton']
price_brighton=brighton_price[['Price']]
"""Roxbury"""
roxbury_price=df_new.loc[df_new['Neighborhood'] == 'Roxbury']
price_roxbury=roxbury_price[['Price']]

"""putting all the prices' dfs in the list"""
price_list=[price_dorchester, price_downtown, price_jamaicaplain, price_brighton, price_roxbury]
# print(price_list)
"""creating an empty list"""
price_list_dist=[]
"""creating list for Neighborhood column"""
neighborhood_list=['Dorchester', 'Downtown', 'Jamaica Plain', 'Brighton', 'Roxbury']
"""creating a for loop to get statistics for price ranges and append it to our empty list"""
for x in price_list:
    i=x.describe(percentiles=[.25, .50, .75])
    i=i.iloc[3:]
    i.reset_index(inplace=True)
    i.rename(columns={'index':'Stats'}, inplace=True)
    price_list_dist.append(i)
"""changing names of the price column to the Borough name"""   
price_list_dist[0].rename(columns={'Price':neighborhood_list[0]}, inplace=True)
price_list_dist[1].rename(columns={'Price':neighborhood_list[1]}, inplace=True)
price_list_dist[2].rename(columns={'Price':neighborhood_list[2]}, inplace=True)
price_list_dist[3].rename(columns={'Price':neighborhood_list[3]}, inplace=True)
price_list_dist[4].rename(columns={'Price':neighborhood_list[4]}, inplace=True)
"""finilizing our dataframe for final view"""    
stat_df=price_list_dist
stat_df=[df.set_index('Stats') for df in stat_df]
stat_df=stat_df[0].join(stat_df[1:])
# print(stat_df)

"""bar graph of boston room type"""
sns.countplot(df_new['Room_Type'], palette="plasma")
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.title('Room Types of Boston')
# plt.show()

"""pie chart of boston room type"""  
# Creating dataset 
Room_Type = ['Entire home/apt', 'Private room', 'Shared room', 'Hotel room'] 
data = [2074, 1139, 14, 27] 
# Creating plot 
fig = plt.figure(figsize =(10, 7)) 
plt.pie(data, labels = Room_Type)
# get percentages
fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
room_data = ["2074 Entire",
          "1139 Private",
          "14 Shared",
          "27 Hotel"]
data = [float(x.split()[0]) for x in room_data]
room_type = [x.split()[-1] for x in room_data]
def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n({:d})".format(pct, absolute)
wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),
                                  textprops=dict(color="w"))
ax.legend(wedges, room_type,
          title="Room Type Legend",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))
plt.setp(autotexts, size=8, weight="bold")
ax.set_title("Pie Chart to show Room Types in Boston")
# plt.show()

"""sort by room type"""
sub_7=df_new.loc[df_new['Neighborhood'].isin(['Dorchester','Downtown','Jamaica Plain','Brighton',
                 'Roxbury','South End','Back Bay','East Boston','Allston','South Boston'])]
viz_3=sns.catplot(x='Neighborhood', col='Room_Type', data=sub_7, kind='count')
viz_3.set_xticklabels(rotation=90) 
# plt.show()

"""correlation"""
corr = df_total.corr(method='kendall')
# corr = df_new.corr(method='kendall')
plt.figure(figsize=(15,8))
sns.heatmap(corr, annot=True)
print(corr)

"""map of neighborhoods"""
plt.figure(figsize=(10,6))
sns.scatterplot(df_new.Longitude,df_new.Latitude,hue=df_new.Neighborhood)
# plt.show()

"""Regression model"""
df_new.drop(['Latitude','Longitude'], axis=1, inplace=True)
#df_new.insert(['availability_365'], )
"""examing the changes"""
#print(df_new)
"""encoding input variables"""
def input_var(data):
    for column in data.columns[data.columns.isin(['Neighborhood', 'Room_Type'])]:
        data[column] = data[column].factorize()[0]
    return data
df_en = input_var(df_new.copy())
#print(df_en.head(10))
"""independent variables and dependent variables"""
x = df_en.iloc[:,[0,1,3,4]]
y = df_en['Price']
"""Getting Test and Training Set"""
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.2,random_state=100)
# print(x_test.shape)
# print(x_train.shape)
# print(y_train.head())

"""prepare a regression model""" 
regression = LinearRegression()
regression.fit(x_train, y_train)
y_pred = regression.predict(x_test) #unsure, what is this trying to do 
#print(r2_score(y_test, y_pred))

"""MAP"""
import folium
"""Create a map:"""
BostonMap = folium.Map(location=[42.3601, -71.0589], width=500, height=500)
"""Create a layer, shaded by neighborhoods:"""
BostonMap.choropleth(geo_data="team_project/neighbourhoods.json",
                     fill_opacity=0.1, line_opacity=0.5, zoom_start=12
                     ) 
"""Output the map to an .html file:"""
print(BostonMap)
BostonMap.save(outfile='team_project/templates/bostonMap.html')
# """Connect df to map"""
# BostonMap.choropleth(geo_data="team_project/neighbourhoods.json",
#                      fill_color='YlGnBu', 
#                      fill_opacity=0.5, 
#                      line_opacity=0.5,
#                     #  threshold_scale = [100,200,300,400],
#                      data = df_new,
#                      key_on='feature.geometry',
#                      columns = ['Neighborhood', 'Price']
#                      ) 
"""Set markers for neighborhoods"""
def set_markers(lat, lon, place, price, nbr):
    coord = [lat, lon]
    folium.Marker(coord, popup=f'{place}, \n {nbr}, \n Average Price: ${price}', tooltip=f"{nbr}", icon=folium.Icon(color='pink')).add_to(BostonMap)
    return BostonMap.save(outfile='team_project/templates/bostonMap.html')

set_markers(42.3527, -71.1106, 'Boston University Bridge', 103.09, 'Allston')
set_markers(42.349396, -71.078369, 'Boston Public Library', 185.71, 'Back Bay')
set_markers(42.3491, -71.0683, 'Bay Village Garden', 123.84, 'Bay Village')
set_markers(42.3588, -71.0638, 'Massachusetts State House', 168.96, 'Beacon Hill')
set_markers(42.3489, -71.1480, 'St.Elizabeth Medical Center', 123.06, 'Brighton')
set_markers(42.3764, -71.0608, 'Bunker Hill Monument', 216.35, 'Charlestown')
set_markers(42.3495, -71.0628, 'Tufts Medical Center', 182.31, 'Chinatown')
set_markers(42.3011, -71.0618, 'Boston Arts Academy', 116.55, 'Dorchester')
set_markers(42.3587, -71.0575, 'Old State House', 218.88, 'Downtown')
set_markers(42.3650, -71.0361, 'Piers Park', 142.28, 'East Boston')
set_markers(42.3394, -71.0940, 'Museum of Fine Arts', 332.23, 'Fenway')
set_markers(42.2615, -71.1364, 'Stony Brook Park', 72.41, 'Hyde Park')
set_markers(42.3074, -71.1208, 'Arnold Arboretum of Harvard University', 141.15, 'Jamaica Plain')
set_markers(42.3601, -71.0589, 'Boston City Hall', 24094.50, 'Leather District')
set_markers(42.3387, -71.0989, 'Isabella Stewart Gardner Museum', 116.91, 'Longwood Medical Area')
set_markers(42.2686, -71.0935, 'Mattapan Square', 79.98, 'Mattapan')
set_markers(42.3326785, -71.0998606,'Boston Basilica of Our Lady of Perpetual Help', 129.05, 'Mission Hill')
set_markers(42.3649708, -71.0570687, 'The Paul Revere House', 160.54, 'North End')
set_markers(42.2714758, -71.1377954, 'George Wright Golf Course', 99.67, 'Roslindale')
set_markers(42.3092075, -71.0910525, 'Franklin Park Zoo', 112.13, 'Roxbury')
set_markers(42.3402396, -71.0511977, 'Boston Convention and Exhibition Center', 178.57, 'South Boston')
set_markers(42.3302697, -71.047082, 'Carson Beach', 206.43, 'South Boston Waterfront')
set_markers(42.3438672, -71.0715579, 'Cathedral of the Holy Cross', 183.39, 'South End')
set_markers(42.3662519, -71.0699514, 'Museum of Science', 256.97, 'West End')
set_markers(42.2842048, -71.1754214, 'Brook Farm', 221.81, 'West Roxbury')
