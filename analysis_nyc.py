import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#%matplotlib inline
import seaborn as sns
from bokeh.models import *
from bokeh.plotting import *
from bokeh.io import *
from bokeh.tile_providers import *
from bokeh.palettes import *
from bokeh.transform import *
from bokeh.layouts import *
from bokeh.plotting import ColumnDataSource, figure, output_file, show
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score

#reading data
data = pd.read_csv(r"team_project/listings.csv") #for when deedee is coding
#data = pd.read_csv(r"listings.csv") #for when julia is coding

#created dataframe of relevant variables
df = pd.DataFrame(data, columns = ['neighbourhood_group','neighbourhood', 'room_type', 'price', 'minimum_nights','number_of_reviews', 'latitude', 'longitude'])
# print(df)

#cleaning/checking data

#checking datatype
#print(df.dtypes)
#checking for null values
#print(df.isnull().sum())
#renaming columns to make more sense
df_new = df.rename(columns={'neighbourhood_group':'Borough','neighbourhood':'Neighborhood','room_type':'Room_Type','price':'Price','minimum_nights':'Minimum_Nights','number_of_reviews':'Number_of_Reviews', 'latitude':"Latitude", 'longitude':'Longitude'})
# print(df_new)

#check for unique values
# print(df_new.Borough.unique())
#because there were so many different neighborhoods, we thought len would be better for analysis
#print(df_new.Neighborhood.unique())
# print(len(df_new.Neighborhood.unique()))
#print(df_new.Room_Type.unique())

#check to see which neighborhoods have the most Airbnb listings. shows top 25 neighborhoods
top_neighborhoods = df_new.Neighborhood.value_counts().head(25)
#print(top_neighborhoods)
#create table to show data (for map analysis later)
top_neighborhoods_df=pd.DataFrame(top_neighborhoods)
top_neighborhoods_df.reset_index(inplace=True)
top_neighborhoods_df.rename(columns={'index':'Neighborhood','Neighborhood':'Number of Listings'}, inplace=True)
# print(top_neighborhoods_df)

#bar graph
neighborhood_bar=sns.barplot(x='Neighborhood', y='Number of Listings',data=top_neighborhoods_df, palette='Greens_d')
neighborhood_bar.set_title('Number of Listings by Neighborhood')
neighborhood_bar.set_xlabel('Neighborhood')
neighborhood_bar.set_ylabel('Number of Listings')
neighborhood_bar.set_xticklabels(neighborhood_bar.get_xticklabels(), rotation=45)
# plt.show()

#price distribution by Neighborhood on a map
#convert latitdue and longitude to mercator values to plot on a map
average_prices_df = pd.read_csv(r"team_project/average_prices_final.csv") #for deedee
#average_prices_df = pd.read_csv(r"Average Prices Final.csv") #for julia

def lat_lon_to_mercator(df, lon, lat):
    """Converts decimal longitude/latitude to Web Mercator format"""
    k = 6378137
    df["x"] = df[lon] * (k * np.pi/180.0)
    df["y"] = np.log(np.tan((90 + df[lat]) * np.pi/360.0)) * k
    return df

mercator_df = lat_lon_to_mercator(average_prices_df,'longitude','latitude')
# print(mercator_df)
#Establishing a zoom scale for the map. 
scale=2000
x=mercator_df['x']
y=mercator_df['y']
#map is automatically centered on the plot elements.
x_min=int(x.mean() - (scale * 100))
x_max=int(x.mean() + (scale * 100))
y_min=int(y.mean() - (scale * 100))
y_max=int(y.mean() + (scale * 100))
#Defining the map tiles to use.
tile_provider=get_provider(OSM)
#Establish the bokeh plot object and add the map tile as an underlay. Hide x and y axis.
plot = figure(
    title='Airbnb Average Prices by NY Neighborhoods',
    match_aspect=True,
    tools='wheel_zoom,pan,reset,save',
    x_range=(x_min, x_max),
    y_range=(y_min, y_max),
    x_axis_type='mercator',
    y_axis_type='mercator'
    )
plot.grid.visible = True
map=plot.add_tile(tile_provider)
map.level='underlay'
plot.xaxis.visible = False
plot.yaxis.visible = False

#function takes scale (defined above), the initialized plot object, and the converted dataframe with mercator coordinates to create a hexbin map
def hex_map(plot,df, scale,leg_label='Hexbin Heatmap'):
#   source = ColumnDataSource(data = mercator_df)
  r,bins=plot.hexbin(x,y,size=scale*1,hover_color='pink',hover_alpha=0.8,legend_label=leg_label)
  hex_hover = HoverTool(tooltips=[('price','@average_price')],mode='mouse',point_policy='follow_mouse',renderers=[r])
  hex_hover.renderers.append(r)
  plot.tools.append(hex_hover)
  
  plot.legend.location = "top_right"
  plot.legend.click_policy="hide"

hex_map(plot=plot,
        df=mercator_df, 
        scale=scale,
        leg_label='Airbnb NY Neighborhoods')

# show(plot)

# #Brooklyn
# brooklyn_price=df_new.loc[df_new['Borough'] == 'Brooklyn']
# price_brooklyn=brooklyn_price[['Price']]
# #Manhattan
# manhattan_price=df_new.loc[df_new['Borough'] == 'Manhattan']
# price_manhattan=manhattan_price[['Price']]
# #Queens
# queens_price=df_new.loc[df_new['Borough'] == 'Queens']
# price_queens=queens_price[['Price']]
# #Staten Island
# statenisland_price=df_new.loc[df_new['Borough'] == 'Staten Island']
# price_statenisland=statenisland_price[['Price']]
# #Bronx
# bronx_price=df_new.loc[df_new['Borough'] == 'Bronx']
# price_bronx=bronx_price[['Price']]
# #putting all the prices' dfs in the list
# price_list=[price_brooklyn, price_manhattan, price_queens, price_statenisland, price_bronx]
# # print(price_list)
# #creating an empty list
# price_list_dist=[]
# #creating list for Borough column
# borough_list=['Brooklyn', 'Manhattan', 'Queens', 'Staten Island', 'Bronx']
# #creating a for loop to get statistics for price ranges and append it to our empty list
# for x in price_list:
#     i=x.describe(percentiles=[.25, .50, .75])
#     i=i.iloc[3:]
#     i.reset_index(inplace=True)
#     i.rename(columns={'index':'Stats'}, inplace=True)
#     price_list_dist.append(i)
# #changing names of the price column to the Borough name   
# price_list_dist[0].rename(columns={'Price':borough_list[0]}, inplace=True)
# price_list_dist[1].rename(columns={'Price':borough_list[1]}, inplace=True)
# price_list_dist[2].rename(columns={'Price':borough_list[2]}, inplace=True)
# price_list_dist[3].rename(columns={'Price':borough_list[3]}, inplace=True)
# price_list_dist[4].rename(columns={'Price':borough_list[4]}, inplace=True)
# #finilizing our dataframe for final view    
# stat_df=price_list_dist
# stat_df=[df.set_index('Stats') for df in stat_df]
# stat_df=stat_df[0].join(stat_df[1:])
# # print(stat_df)

# sort by room type
# sub_7=df_new.loc[df_new['Neighbourhood'].isin(['Williamsburg','Bedford-Stuyvesant','Harlem','Bushwick',
#                  'Upper West Side','Hell\'s Kitchen','East Village','Upper East Side','Crown Heights','Midtown'])]
# viz_3=sns.catplot(x='Neighborhood', hue='Borough', col='Room_Type', data=sub_7, kind='count')
# viz_3.set_xticklabels(rotation=90) 
# print(viz_3)

# # map of borough
# plt.figure(figsize=(10,6))
# sns.scatterplot(df_new.Longitude,df_new.Latitude,hue=df_new.Borough)
# plt.ioff()

# Regression model
df_new.drop(['Latitude','Longitude'], axis=1, inplace=True)
#examing the changes
# print(df_new)
# encoding input variables
def input_var(data):
    for column in data.columns[data.columns.isin(['Borough', 'Room_Type', 'Neighborhood'])]:
        data[column] = data[column].factorize()[0]
    return data
df_en = input_var(df_new.copy())
# print(df_en.head(20))
#Get Correlation between different variables
corr = df_en.corr(method='kendall')
plt.figure(figsize=(18,12))
sns.heatmap(corr, annot=True)
# plt.show()
#independent variables and dependent variables
x = df_en.iloc[:,[0,1,2,4,5]]
y = df_en['Price']
#Getting Test and Training Set
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.2,random_state=1000)
print(x_test.shape)
print(x_train.shape)
# print(y_train.head())

#prepare a regression model 
regression = LinearRegression()
regression.fit(x_train, y_train)
y_pred = regression.predict(x_test) #unsure, what is this trying to do 
print(r2_score(y_test, y_pred))
