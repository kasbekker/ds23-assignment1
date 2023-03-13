
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
    p = figure(title = 'Sales Volume per country in the second half of 2021', plot_height = 600 , plot_width = 950, tools=['xpan', 'reset', 'save', 'wheel_zoom', hover])
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
  p = figure(title = 'Average rating per country for the second half of 2021', plot_height = 600 , plot_width = 950,  tools = ['xpan', 'reset', 'save', 'wheel_zoom', hover])
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

def ratings():
  #data handling
  import pandas as pd
  import numpy as np
  import glob, os

  # Bokeh libraries
  from bokeh.io import output_file, output_notebook
  from bokeh.plotting import figure, show
  from bokeh.models import ColumnDataSource
  from bokeh.layouts import row, column, gridplot
  from bokeh.io import output_file
  from bokeh.plotting import figure, show
  from bokeh.models import HoverTool
  from bokeh.models import DatetimeTickFormatter
  from bokeh.models import DataRange1d

  # %%
  # Get CSV files list from a folder
  path = 'data/ratings_country_data'
  csv_files = glob.glob(path + "/*.csv")

  # Read each CSV file into DataFrame
  # This creates a list of dataframes
  df_list = (pd.read_csv(file, encoding = 'utf-16') for file in csv_files)

  #encoding moest hier gedefinieerd worden met 'utf-16' anders kreeg ik een error, https://stackoverflow.com/questions/50342517/unicodedecodeerror-reading-a-csv-file

  # Concatenate all DataFrames
  df = pd.concat(df_list, ignore_index=True)

  #Bovenstaande code selecteert een map in path, en zet alle csv bestanden in een dataframe genaamd df, https://sparkbyexamples.com/pandas/pandas-read-multiple-csv-files/

  print(df)

  # %%
  #Selecteert de rijen die van belang zijn (zie opdracht document)
  df = df[["Date", "Package Name", "Country", "Daily Average Rating", "Total Average Rating"]]

  print(df)

  # %%
  #Laat het percentage NaN values zien.
  df.isna().sum()/len(df)*100

  # %%
  #Dropt alle rijen waar NaN in voorkomt, https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html
  df = df.dropna()

  print(df)

  # %%
  #Laat zien dat er nu geen 1 rij NaN type heeft. 
  df.isna().sum()/len(df)*100

  # %%
  #Vertaalt de data in de kolom 'date' naar leesbaar data voor de pandas dataframe  
  df['Date'] = pd.to_datetime(df['Date'])

  df_ratings = df

  print(df_ratings)

  # %%
  #selecteerd de kolommen 'date' en 'Daily Average Rating'
  df_date_rating = df_ratings[["Date", "Daily Average Rating"]]

  #Maakt cds bestanden en zorgt ervoor dat 'Date' een kolom blijft in de dataframe
  df_date_rating_cds = ColumnDataSource(df_date_rating.groupby('Date')['Daily Average Rating'].sum().to_frame().reset_index())

  print(df_date_rating)

  # %%
  #Geeft de gemiddelde average rating per maand
  df_date_avg_rating_monthly = df_date_rating.groupby(pd.Grouper(key='Date', freq='M')).mean().reset_index()  

  print(df_date_avg_rating_monthly)

  # %%
  #Zet het figuur op
  fig = figure(title = 'Average rating per month', 
              x_axis_type='datetime', 
              height=400, 
              width=800, 
              x_axis_label='Date', 
              y_axis_label='Rating')

  source = ColumnDataSource(df_date_avg_rating_monthly)

  # Add the hover tool to the figure
  hover = HoverTool(tooltips=[('Rating', '@{Daily Average Rating}')])

  fig.vbar(x='Date', 
          top='Daily Average Rating', 
          source=source, width=86400000*28)

  # Add the hover tool to the figure
  fig.add_tools(hover)
  return(fig)



def crashes():
  # %%
  #data handling
  import pandas as pd
  import numpy as np
  import glob, os

  # Bokeh libraries
  from bokeh.io import output_file, output_notebook
  from bokeh.plotting import figure, show
  from bokeh.models import ColumnDataSource
  from bokeh.io import output_file
  from bokeh.plotting import figure, show
  from bokeh.models import HoverTool
  from bokeh.models import DatetimeTickFormatter

  # %%
  # Get CSV files list from a folder
  path = 'data/crashes_data'
  csv_files = glob.glob(path + "/*.csv")

  # Read each CSV file into DataFrame
  # This creates a list of dataframes
  df_list = (pd.read_csv(file, encoding = 'utf-16') for file in csv_files)


  #encoding moest hier gedefinieerd worden anders kreeg ik een error, https://stackoverflow.com/questions/50342517/unicodedecodeerror-reading-a-csv-file

  # Concatenate all DataFrames
  df   = pd.concat(df_list, ignore_index=True)


  # %%
  #selecteert de rijen die van belang zijn (zie opdracht document)
  df = df[["Date","Package Name","Daily Crashes","Daily ANRs"]]


  # %%
  #laat het percentage NaN values zijn.
  df.isna().sum()/len(df)*100

  # %%
  #Vertaalt de data in de kolom 'date' naar leesbaar data voor de pandas dataframe 
  df['Date'] = pd.to_datetime(df['Date'])
  
  df_crashes_ANRs = df


  # %%
  #selecteerd de kolommen 'date' en 'Daily Crashes'
  df_date_crashes = df_crashes_ANRs[["Date", "Daily Crashes"]]

  #Maakt cds bestanden en zorgt ervoor dat 'Date' een kolom blijft in de dataframe
  df_date_crashes_cds = ColumnDataSource(df_crashes_ANRs.groupby('Date')['Daily Crashes'].sum().to_frame().reset_index())

  # %%
  #Geeft de gemiddelde average crashes per maand
  df_date_avg_crashes_monthly = df_date_crashes.groupby(pd.Grouper(key='Date', freq='M')).mean().reset_index()


  # %%
  #Figuur opzetten
  fig = figure(title = 'Average crashes per month', 
              x_axis_type='datetime', 
              height=400, 
              width=800, 
              x_axis_label='Date', 
              y_axis_label='Crashes')


  source = ColumnDataSource(df_date_avg_crashes_monthly)

  #Add the hover tool to the figure
  hover = HoverTool(tooltips=[('Crashes', '@{Daily Crashes}')])

  fig.vbar(x='Date', top='Daily Crashes', source=source, width=86400000*28)

  # Add the hover tool to the figure
  fig.add_tools(hover)
  #x-as klopt nog niet, hij begint 1 maand later dan in de dataframe!


  # %%
  #selecteerd de kolommen 'date' en 'Daily ANRs'
  df_date_ANRs = df_crashes_ANRs[["Date", "Daily ANRs"]]

  #Maakt cds bestanden
  df_date_ANRs_cds = ColumnDataSource(df_crashes_ANRs.groupby('Date')['Daily ANRs'].sum().to_frame().reset_index())



  # %%
  #Geeft de gemiddelde average ANRs per maand
  df_date_avg_ANRs_monthly = df_date_ANRs.groupby(pd.Grouper(key='Date', freq='M')).mean().reset_index()


  # %%
  #Zet het figuur op
  fig2 = figure(title = 'Average ANRs per month', 
              x_axis_type='datetime', 
              height=400, 
              width=800, 
              x_axis_label='Date', 
              y_axis_label='ANRs')


  source = ColumnDataSource(df_date_avg_ANRs_monthly)

  # Add the hover tool to the figure
  hover = HoverTool(tooltips=[('ANRs', '@{Daily ANRs}')])

  fig2.vbar(x='Date', top='Daily ANRs', source=source, width=86400000*28)

  # Add the hover tool to the figure
  fig2.add_tools(hover)
  return(fig,fig2)

def sales_volume():
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

  def to_integer(dt_time):
      return 10000*dt_time.year + 100*dt_time.month + dt_time.day


  # %%
  import pandas as pd
  import numpy as np
  import glob, os

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
  df.isna().sum()/len(df)*100

  #laat het percentage NaN values zijn.


  # %%
  df = df.dropna()

  #dropt alle rijen waar NaN in voorkomt, https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html



  # %%
  df.isna().sum()/len(df)*100

  #laat zien dat er nu geen 1 rij NaN type heeft. 


  # %%
  df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
  df_sales =df
  #vertaalt de transaction date naar leesbaar data voor pandas dataframe.


  # %%


  # %%
  # Determine where the visualization will be rendered
  # output_file('filename.html')  # Render to static HTML, or 
  output_notebook()  # Render inline in a Jupyter Notebook

  #selecteert de rijen transaction date en amount
  date_amount_data = df_sales[["Transaction Date", "Amount (Merchant Currency)"]]
  date_amount_data.columns = ['Date', 'Amount'] 
  # amount_data = df_sales["Amount (Merchant Currency)"]

  dates = np.array(AAPL['date'], dtype=np.datetime64)
  #maakt cds bestanden voor sales per dag
  # date_data_cds = ColumnDataSource(date_amount_data.groupby('Transaction Date')['Amount (Merchant Currency)'].sum().to_frame().reset_index(), data=dict(date=dates, close=AAPL['adj_close']))
  date_data_cds = ColumnDataSource(date_amount_data.groupby('Date')['Amount'].sum().to_frame().reset_index())
  date_data_cds_count = ColumnDataSource(date_amount_data.groupby(['Date'])['Amount'].count().to_frame().reset_index())
  # amount_data_cds = ColumnDataSource(amount_data)




  # %%
  # Set up the figure(s)
  # Create and configure the figure
  from bokeh.models import HoverTool
  #add hovertool
  hover = HoverTool(tooltips = [ ('Date','@Date'),('Total Sum of Sales', '@Amount')])


  fig = figure(x_axis_type='datetime',
              plot_height=800, plot_width=1500, tools=['xpan', 'reset', 'save', 'wheel_zoom', hover],
              title='Sales volume (EUR) per day in the second half of 2021',
              x_axis_label='Date', y_axis_label='Amount')
  # Connect to and draw the data
  # Render the race as step lines


  #maak vbar fig
  fig.vbar(x='Date', top='Amount', source=date_data_cds, width=8640000*4)

  fig2 = figure(x_axis_type='datetime',
              plot_height=800, plot_width=1500, tools=['xpan', 'reset', 'save', 'wheel_zoom', hover],
              title='Amount of buyers per day in the second half of 2021',
              x_axis_label='Date', y_axis_label='Amount')
  fig2.vbar(x='Date', top='Amount', source=date_data_cds_count, width=8640000*4)

  # %%
  # xdate1 = datetime.date(2018, 6, 1)
  # xdate2 = datetime.date(2018, 6, 3)
  # print(xdate1,xdate2)

  # p = figure(height=300, width=800, tools="xpan", toolbar_location=None,
  #            x_axis_type="datetime", x_axis_location="above",
  #            background_fill_color="#efefef", x_range=(0,5))

  # p.line(x='Transaction Date', y='Amount (Merchant Currency)', source=date_data_cds)


  # select = figure(title="Drag the middle and edges of the selection box to change the range above",
  #                 height=130, width=800, y_range=p.y_range,
  #                 x_axis_type="datetime", y_axis_type=None,
  #                 tools="", toolbar_location=None, background_fill_color="#efefef")

  # range_tool = RangeTool(x_range=p.x_range)
  # range_tool.overlay.fill_color = "navy"
  # range_tool.overlay.fill_alpha = 0.2

  # select.line(x= 'Transaction Date', y= 'Amount (Merchant Currency)', source=date_data_cds)
  # select.ygrid.grid_line_color = None
  # select.add_tools(range_tool)
  # select.toolbar.active_multi = range_tool

  # show(column(p, select))
  # output_file('test.html')



  # %%


  # #range slider
  # select = figure(title="Drag the middle and edges of the selection box to change the range above",
  #                 height=130, width=800, y_range=fig.y_range,
  #                 x_axis_type="datetime", y_axis_type=None,
  #                 tools="", toolbar_location=None, background_fill_color="#efefef")

  # range_tool = RangeTool(x_range=Range1d(0,5))
  # range_tool.overlay.fill_color = "navy"
  # range_tool.overlay.fill_alpha = 0.2

  # select.vbar(x='Transaction Date', top='Amount (Merchant Currency)', source=date_data_cds, width=20)
  # select.ygrid.grid_line_color = None
  # select.add_tools(range_tool)
  # select.toolbar.active_multi = range_tool

  # # Organize the layout

  # # Preview and save 
  # show(column(fig,select))  # See what I made, and save if I like it
  # output_file('test.html')
  return(fig,fig2)

def sku_id():
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

  # Get CSV files list from a folder
  path = 'data/sales_data'
  csv_files = glob.glob(path + "/*.csv")

  # Read each CSV file into DataFrame
  # This creates a list of dataframes
  df_list = (pd.read_csv(file) for file in csv_files)

  # Concatenate all DataFrames
  df   = pd.concat(df_list, ignore_index=True)
  df.head()


  #Bovenstaande code selecteert een map in path, en zet alle csv bestanden in een dataframe genaamd df, https://sparkbyexamples.com/pandas/pandas-read-multiple-csv-files/

  # %%
  df = df[["Product id", "Sku Id", "Product Title", "Amount (Merchant Currency)", "Transaction Type"]]
  #selecteert de rijen die van belang zijn (zie opdracht document)


  df = df.loc[(df['Transaction Type'] == 'Charge')]
  #selecteert specfieke rijen met de waarden die overeenkomen hier boven (zie opdracht document)



  # %%
  df.isna().sum()/len(df)*100

  #laat het percentage NaN values zijn.


  # %%
  df = df.dropna()

  #dropt alle rijen waar NaN in voorkomt, https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html



  # %%
  df.isna().sum()/len(df)*100

  #laat zien dat er nu geen 1 rij NaN type heeft. 


  # %%

  #vertaalt de transaction date naar leesbaar data voor pandas dataframe.
  df.head()


  # %%
  # Determine where the visualization will be rendered
  # output_file('filename.html')  # Render to static HTML, or 
  output_notebook()  # Render inline in a Jupyter Notebook

  #selecteert de rijen transaction date en amount
  # amount_data = df_sales["Amount (Merchant Currency)"]


  #maakt cds bestanden voor sales per dag
  # date_data_cds = ColumnDataSource(date_amount_data.groupby('Transaction Date')['Amount (Merchant Currency)'].sum().to_frame().reset_index(), data=dict(date=dates, close=AAPL['adj_close']))
  df_grouped_sku_id = df.groupby(['Sku Id'])['Amount (Merchant Currency)'].sum().to_frame().reset_index()
  df_grouped_sku_id.columns = ['sku_id', 'amount']
  sku_id_volume_cds = ColumnDataSource(df_grouped_sku_id)
  # amount_data_cds = ColumnDataSource(amount_data)
  df_grouped_sku_id.head()

  # %%
  # Set up the figure(s)
  # Create and configure the figure
  from bokeh.models import HoverTool
  #add hovertool
  hover = HoverTool(tooltips = [ ('Sku ID','@sku_id'),('Total Sum of Sales', '@amount')])
  labels = df_grouped_sku_id['sku_id'].tolist()
  amount = df_grouped_sku_id['amount'].tolist()

  fig = figure(plot_height=800, plot_width=1500, tools=['xpan', 'reset', 'save', 'wheel_zoom', hover],
              title='Sales volume (EUR) per SKU ID in the second half of 2021', x_axis_type='auto',
              x_axis_label='Sku ID', y_axis_label='Amount', x_range=labels)
  # Connect to and draw the data
  # Render the race as step lines
  #maak vbar fig

  fig.vbar(x='sku_id', top='amount', source=sku_id_volume_cds, width=0.5)
  return(fig)


sales = sales_volume()



crashes = crashes()
output_file('final.html')  # Render to static HTML, or 
show(column(sales[0],sales[1],sku_id(), ratings(),crashes[0],crashes[1], sales_volume_per_country(),ratings_per_country()))