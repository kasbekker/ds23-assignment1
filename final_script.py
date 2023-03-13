
# %%
"""Bokeh Visualization Template

This template is a general outline for turning your data into a 
visualization using Bokeh.
"""
# Data handling
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import glob, os
from datetime import date
import datetime

import geopandas as gpd

# Bokeh libraries
from bokeh.io import output_file, output_notebook
from bokeh.plotting import figure, show
from bokeh.layouts import row, column, gridplot
from bokeh.models.widgets import Tabs, Panel
from bokeh.io import output_file
# Bokeh Libraries
from bokeh.models import ColumnDataSource, CategoricalColorMapper, Div, RangeTool, Range1d, CustomJS, DateRangeSlider
from bokeh.sampledata.stocks import AAPL
from bokeh.plotting import figure
from bokeh.resources import CDN
from bokeh.embed import file_html

# %%
import pandas as pd
import numpy as np
import glob, os

def sales_volume_per_country():
    # Get CSV files list from a folder
    path = 'data/sales_data'
    csv_files = glob.glob(path + "/*.csv")

    # Read each CSV file into DataFrame
    # This creates a list of dataframes
    df_list = (pd.read_csv(file) for file in csv_files)

    # Concatenate all DataFrames
    df   = pd.concat(df_list, ignore_index=True)

    #Bovenstaande code selecteert een map in path, en zet alle csv bestanden in een dataframe genaamd df, https://sparkbyexamples.com/pandas/pandas-read-multiple-csv-files/




    # %%
    df = df[["Transaction Date", "Transaction Type", "Product id", "Sku Id", "Buyer Country", "Buyer Postal Code", "Amount (Merchant Currency)"]]
    #selecteert de rijen die van belang zijn (zie opdracht document)


    df = df.loc[(df['Transaction Type'] == 'Charge') & (df['Product id'] == 'com.vansteinengroentjes.apps.ddfive')]
    #selecteert specfieke rijen met de waarden die overeenkomen hier boven (zie opdracht document)



    # %%
    df = df.dropna()

    #dropt alle rijen waar NaN in voorkomt, https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html


    # %%
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
    df_sales =df
    #vertaalt de transaction date naar leesbaar data voor pandas dataframe.



    # %%
    # Determine where the visualization will be rendered
    # output_file('filename.html')  # Render to static HTML, or 

    #selecteert de rijen transaction date en amount
    date_amount_data = df_sales[["Buyer Country", "Amount (Merchant Currency)"]]

    #telt amount bij elkaar op per land
    date_amount_data = date_amount_data.groupby('Buyer Country')['Amount (Merchant Currency)'].sum().to_frame().reset_index()
    date_amount_data.columns = ['Buyer Country','Amount']
    date_amount_data.head()


    # %%
    #geopandas
    shapefile = 'ne_110m_admin_0_countries.shp'
    gdf = gpd.read_file(shapefile)[['ADMIN', 'WB_A2', 'geometry']]
    gdf.columns = ['country', 'country_code', 'geometry'] 

    #Drop row corresponding to 'Antarctica'
    gdf = gdf.drop(gdf.index[159])


    gdf.head()


    # %%
    #Merge dataframes gdf and df_2016.
    merged = gdf.merge(date_amount_data, left_on = 'country_code', right_on = 'Buyer Country',how = 'left')
    #Replace NaN values to string 'No data'.
    merged.fillna('No data', inplace = True)
    merged.head()

    # %%
    import json
    #Read data to json.
    merged_json = json.loads(merged.to_json())
    #Convert to String like object.
    json_data = json.dumps(merged_json)



    # %%
    from bokeh.io import output_notebook, show, output_file
    from bokeh.plotting import figure
    from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar, HoverTool
    from bokeh.palettes import brewer

    #Input GeoJSON source that contains features for plotting.
    geosource = GeoJSONDataSource(geojson = json_data)

    #Define a sequential multi-hue color palette.
    palette = brewer['YlGnBu'][8]

    #Reverse color order so that dark blue is highest obesity.
    palette = palette[::-1]

    #Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.

    color_mapper = LinearColorMapper(palette = palette, low = 0, high = 100, nan_color = '#d9d9d9')
    #Define custom tick labels for color bar.
    tick_labels =  {'0': '0', '10': '10', '20':'20', '30':'30', '40':'40', '50':'50', '60':'60','70':'70', '80': '80', '90': '90', '100': '>100'}
    # tick_labels = {'0': '0', '5': '5', '10':'10', '15':'15', '20':'20','25': '25','30': '30','35': '35','40': '40','45': '45','50': '>50'}

    #add hovertool
    hover = HoverTool(tooltips = [ ('Country/region','@country'),('Total Sum of Sales', '@Amount')])

    #Create color bar. 
    color_bar = ColorBar(color_mapper=color_mapper, label_standoff=8,width = 500, height = 20,
    border_line_color=None,location = (0,0), orientation = 'horizontal', major_label_overrides = tick_labels)

    #Create figure object.
    p = figure(title = 'Sales Volume per country in the second half of 2021', plot_height = 600 , plot_width = 950, toolbar_location = None, tools=[hover])
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    #Add patch renderer to figure. 
    p.patches('xs','ys', source = geosource, fill_color = {'field' :'Amount', 'transform' : color_mapper},
            line_color = 'black', line_width = 0.25, fill_alpha = 1)

    #Specify figure layout.
    p.add_layout(color_bar, 'below')
    #Display figure inline in Jupyter Notebook.
    #Display figure.
    return(p)

def ratings_per_country():
  # Get CSV files list from a folder
  path = 'data/ratings_country_data'
  csv_files = glob.glob(path + "/*.csv")

  # Read each CSV file into DataFrame
  # This creates a list of dataframes
  df_list = (pd.read_csv(file,  encoding="utf-16" ) for file in csv_files)

  # Concatenate all DataFrames
  df   = pd.concat(df_list, ignore_index=True)

  #Bovenstaande code selecteert een map in path, en zet alle csv bestanden in een dataframe genaamd df, https://sparkbyexamples.com/pandas/pandas-read-multiple-csv-files/
  df = df.reset_index()

  df.head()


  # %%
  df = df[["Date", "Package Name", "Country", "Total Average Rating", "index"]]
  #selecteert de rijen die van belang zijn (zie opdracht document)


  df = df.loc[(df['Package Name'] == 'com.vansteinengroentjes.apps.ddfive')]
  #selecteert specfieke rijen met de waarden die overeenkomen hier boven (zie opdracht document)
  df.head()


  # %%
  df = df.dropna()

  #dropt alle rijen waar NaN in voorkomt, https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html
  df.head()

  # %%
  df['Date'] = pd.to_datetime(df['Date'])
  df_sales_ratings =df
  #vertaalt de transaction date naar leesbaar data voor pandas dataframe.



  # %%
  # Determine where the visualization will be rendered
  # output_file('filename.html')  # Render to static HTML, or 

  #selecteert de rijen 
  country_average_df = df_sales_ratings[["Country", "Total Average Rating"]]

  #neemt average rating per country
  country_average_df = country_average_df.groupby('Country')['Total Average Rating'].mean().to_frame().reset_index()
  country_average_df.columns = ['Country','total_avg_rating']
  country_average_df.head()


  # %%
  #geopandas
  shapefile = 'ne_110m_admin_0_countries.shp'
  gdf = gpd.read_file(shapefile)[['ADMIN', 'WB_A2', 'geometry']]
  gdf.columns = ['country', 'country_code', 'geometry'] 

  #Drop row corresponding to 'Antarctica'
  gdf = gdf.drop(gdf.index[159])

  gdf.head()

  # %%
  #Merge dataframes gdf and df_sales.
  merged = gdf.merge(country_average_df, left_on = 'country_code', right_on = 'Country',how = 'left')
  #Replace NaN values to string 'No data'.
  merged.fillna('No data', inplace = True)
  merged.head()

  # %%
  import json
  #Read data to json.
  merged_json = json.loads(merged.to_json())
  #Convert to String like object.
  json_data = json.dumps(merged_json)
  # %%
  from bokeh.io import output_notebook, show, output_file
  from bokeh.plotting import figure
  from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar, HoverTool
  from bokeh.palettes import brewer

  #Input GeoJSON source that contains features for plotting.
  geosource = GeoJSONDataSource(geojson = json_data)

  #Define a sequential multi-hue color palette.
  palette = brewer['YlGnBu'][8]

  #Reverse color order so that dark blue is highest obesity.
  palette = palette[::-1]

  #Add hover tool
  hover = HoverTool(tooltips = [ ('Country/region','@country'),('Total Average Rating', '@total_avg_rating')])

  #Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.

  color_mapper = LinearColorMapper(palette = palette, low = 0, high = 5, nan_color = '#d9d9d9')
  #Define custom tick labels for color bar.
  tick_labels_100= {'0': '0', '10': '10', '20':'20', '30':'30', '40':'40', '50':'50', '60':'60','70':'70', '80': '80', '90': '90', '100': '>100'}
  tick_labels = {'0': '0', '5': '5', '10':'10', '15':'15', '20':'20','25': '25','30': '30','35': '35','40': '40','45': '45','50': '>50'}

  #Create color bar. 
  color_bar = ColorBar(color_mapper=color_mapper, label_standoff=8,width = 500, height = 20,
  border_line_color=None,location = (0,0), orientation = 'horizontal', major_label_overrides = tick_labels)

  #Create figure object.
  p = figure(title = 'Average rating per country for the second half of 2021', plot_height = 600 , plot_width = 950, toolbar_location = None,  tools = [hover])
  p.xgrid.grid_line_color = None
  p.ygrid.grid_line_color = None

  #Add patch renderer to figure. 
  p.patches('xs','ys', source = geosource, fill_color = {'field' :'total_avg_rating', 'transform' : color_mapper},
            line_color = 'black', line_width = 0.25, fill_alpha = 1)

  #Specify figure layout.
  p.add_layout(color_bar, 'below')
  #Display figure inline in Jupyter Notebook.
  #Display figure.
  return(p)
    
output_file('final.html')  # Render to static HTML, or 
show(column(sales_volume_per_country(),ratings_per_country()))