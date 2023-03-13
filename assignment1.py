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

# Bokeh libraries
from bokeh.io import output_file, output_notebook
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
from bokeh.layouts import row, column, gridplot
from bokeh.models.widgets import Tabs, Panel

# Prepare the data

# Get CSV files list from a folder
def get_df_sales():
  path = 'data/sales_data'
  csv_files = glob.glob(path + "/*.csv")

  # Read each CSV file into DataFrame
  # This creates a list of dataframes
  df_list = (pd.read_csv(file) for file in csv_files)

  # Concatenate all DataFrames
  df   = pd.concat(df_list, ignore_index=True)
  df = df[["Transaction Date", "Transaction Type", "Product id", "Sku Id", "Buyer Country", "Buyer Postal Code", "Amount (Merchant Currency)"]]
  #selecteert de rijen die van belang zijn (zie opdracht document)

  df.loc[(df['Transaction Type'] == 'Charge') & (df['Product id'] == 'com.vansteinengroentjes.apps.ddfive')]
  #selecteert specfieke rijen met de waarden die overeenkomen hier boven (zie opdracht document)

  df = df.dropna() #dropt alle NaN rijen
  df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
  return df

df_sales = get_df_sales()

#visaluseer de sales over tijd (per  maand/dag of) in termen van ten minste twee meet waarden Amount of Row count












# Determine where the visualization will be rendered
output_file('filename.html')  # Render to static HTML, or 
output_notebook()  # Render inline in a Jupyter Notebook

# Set up the figure(s)
fig = figure()  # Instantiate a figure() object

# Connect to and draw the data

# Organize the layout

# Preview and save 
show(fig)  # See what I made, and save if I like it