import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import base64
from matplotlib.pyplot import imread
import plotly.express as px
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from io import BytesIO
from wordcloud import WordCloud, STOPWORDS
import plotly.graph_objs as go
import io
import datetime

mapbox_access_token = 'pk.eyJ1Ijoia2Nsb3c5NyIsImEiOiJja2ViNWwzcDEwNWxvMnJwOHgwcXI1b3lmIn0.ESka3oeP-Tw-wbEpJfsiww'

external_scripts = ['/assets/style.css']

app = dash.Dash(__name__,
                external_scripts=external_scripts)
server = app.server

df_original = pd.read_csv("Housing_data.csv")
df = df_original.copy()
dfi = df_original.copy()
columns = df_original.columns
df.columns = columns
dfi.columns = columns

df_count = dfi.copy()
df_count['计数'] = 1

dfcorr = df[['Price', 'Built_year', 'Price per Sqft', 'Bathroom', 'Bedroom']]
#calculate relationship between attributes
corr = dfcorr.corr(method = 'pearson').round(3)

def make_dash_table(df):
    table = []
    html_col_name = ['-']
    for col in df.columns:
        html_col_name.append(html.Td([col]))
    table.append(html.Tr(html_col_name))
    for index, row in df.iterrows():
        html_row = [index]
        for i in range(len(row)):
            html_row.append(html.Td([row[i]]))
        table.append(html.Tr(html_row))
    return table

blackbold={'color':'black', 'font-weight': 'bold'}

pie_list = ['State', 'Category', 'Ownership', 'Furnishing', 'Floor Level']
type_list = ['Number', 'Ratio']

data = pd.DataFrame()
data1 = pd.DataFrame()
data = df['Room_type'].value_counts()
data1 = data[data>=300]

area = df.groupby(['State']).mean()
unitprice = area['Price'].round().sort_values()
mean_graph_x = unitprice.index

state_list = df["State"].unique()

fig_row5 = px.scatter(df, x="Square_feet", y="Price",
                 size="Price per Sqft", color="Category", hover_name="Type",
                 log_x=True, size_max=60)
fig_row5.update_yaxes(nticks=20)

df_row6 = px.data.gapminder()
fig_row6 = px.scatter(df, x = "Price", y = "Square_feet",
                      animation_frame = "Built_year", animation_group = "Category",
                      size="Price per Sqft", color="Type", hover_name="Type", facet_col="Category",
                      log_x=True, size_max=45,
                      range_x=[45000,1000000],
                      range_y=[100, 4000])

df2 = df[df.Square_feet <= 10000]
fig_row7a = px.scatter(df2,  x = "Square_feet", y = "Price")
fig_row7a.update_yaxes(title_text = 'Housing Price (MYR)', nticks=20,
                        title_font=dict(size=18, family='Courier', color='crimson')
                       )
fig_row7a.update_xaxes(title_text = 'Square Feet (m²)',
                        title_font=dict(size=18, family='Courier', color='crimson')
                       )

fig_row7b = px.scatter(df,  x = "Square_feet", y = "Price per Sqft")
fig_row7a.update_yaxes(title_text = 'Housing Price (MYR)', nticks=20,
                        title_font=dict(size=18, family='Courier', color='crimson')
                       )
fig_row7a.update_xaxes(title_text = 'Square Feet (m²)',
                        title_font=dict(size=18, family='Courier', color='crimson')
                       )

backpicture = "house1.jpg"
color_mask = imread(backpicture)
def plot_wordcloud():
    comment_words = ''
    stopwords = set(STOPWORDS)

    for val in df.Name:
        val = str(val)
        tokens = val.split()
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        comment_words += " ".join(tokens) + " "

    wordcloud = WordCloud(font_path = "msyh.ttc",
                          mask = color_mask,
                          contour_color = 'steelblue',
                          background_color="rgba(255, 255, 255, 0)",
                          stopwords = stopwords,
                          max_words = 2000,
                          max_font_size = 60).generate(comment_words)
    return wordcloud.to_image()

df_prediction = pd.read_csv("Housing_data_predict.csv")

# Use below four variables as predict variables
cols = ['Bedroom', 'Bathroom', 'Square_feet', 'Price per Sqft']
x = df_prediction[cols].values
# Use price as response variable
y = df_prediction['Price'].values

X_train, X_test, y_train, y_test = train_test_split(
       x, y, test_size = 0.33, random_state = 42
)

clf = RandomForestRegressor(n_estimators = 400)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#Score model
rfr = clf
testing_data = df_prediction[cols]
final_test = testing_data.values
y_te_pred = rfr.predict(final_test)
prediction = pd.DataFrame(y_te_pred, columns=['Predicted Price'])
prediction_result = pd.concat([ df_prediction['ID'], df_prediction['Type'],df_prediction['Bedroom'],
                                df_prediction['Bathroom'], df_prediction['Square_feet'],
                                df_prediction['Price per Sqft'], df_prediction['Price'], prediction], axis=1)
prediction_result = prediction_result.astype({"Predicted Price": int})
test_mae = metrics.mean_absolute_error(y_test, y_pred)

# Actual vs Predicted Housing Price Figure
avp_fig = px.scatter(
    x=y_test,
    y=y_pred,
    labels = {"x": "Actual", "y": "Predicted"},
    title = f"Actual vs predicted interest rate (MAE={test_mae:.2f})",
)
avp_fig.add_shape(
    type = "line",
    x0 = y_test.min(), y0 = y_test.min(),
    x1 = y_test.max(), y1 = y_test.max()
)

colors = {"graphBackground": "#F5F5F5", "background": "#ffffff", "text": "#000000"}

app.layout = html.Div([
    html.H1("HOUSING DATA ANALYTICS", style={"textAlign": "center", "color": "black",
                                            "font-family": 'Open Sans', "font-size": '4rem',
                                            "font-weight": "800", "text-transform:": "uppercase",
                                             "line-height": "normal",
                                             }),
        dcc.Markdown('''• This is an interactive visualization dashboard, to provide 
        users an experience of housing market in Malaysia in a infographic manners. '''),
        dcc.Markdown('''• Data is acquired from PropertyGuru Malaysia'''),
    html.Br([]),

    dcc.Tabs(id = "tabs", children=[
    dcc.Tab(label = 'Housing Prices',
            style={"textAlign": "center",
                   "color": "#0000A0",
                    "font-family": 'Open Sans', "font-size": '3rem',
                    "font-weight": "800", "text-transform:": "uppercase",
                    "line-height": "normal",
                    },
            children=[
                html.Div([html.H3("Dataset Introduction", style={'textAlign': 'center'}),
                          html.Br([]),
                          dash_table.DataTable(
                              id='dataset-intro-table',
                              columns=[{'name': i, 'id': i}
                                       for i in df.loc[:,
                                                ['State', 'Location', 'Price', 'Type', 'Built_year', 'Room_type',
                                                 'Furnishing']]],
                              data=df[0:5].to_dict('rows'),
                              style_cell={'textAlign': 'left', 'backgroundColor': '#92a8d1', 'color': 'white'},
                              style_header={'border': '1px solid blue', 'fontWeight': 'bold'},
                              style_data={'border': '1px solid pink'},
                              style_cell_conditional=[
                                  {
                                      'if': {'column_id': 'Region'},
                                      'textAlign': 'left'
                                  }
                              ]
                          ),

    html.H3("A Glance of Housing Price in Malaysia", style={'textAlign': 'center', 'padding-top': -10}),
        dcc.Graph(
            id='mean graph',
            figure={
                'data': [
                    {'x': mean_graph_x,
                     'marker': {'color': ['#000080','#342D7E','#15317E','#0000A0',
                                          '#0020C2','#0041C2','#2554C7','#1569C7',
                                          '#2554C7','#2B60DE','#1F45FC','#306EFF',
                                          '#1589FF','#6495ED','#6698FF','#38ACEC'
                                          '#82CAFA','#82CAFF',
                                          ]},
                     'y': unitprice, 'type': 'bar'},
                     ],
                'layout': {
                    'xaxis': {
                        'type': 'category',
                        'title': 'States',
                    },
                    'yaxis': {
                        'range': [0, 800000],
                        'title': 'Price (MYR)'
                    },
                    'title': 'Mean of Housing Price in Malaysia',
                }
            }
        ),

    html.H3("House Price Versus Square Feet", style={'textAlign': 'center', 'padding-top': 5}),
            dcc.Graph(
                id='price vs sqft',
                figure = fig_row5
            ),

    html.H3("House Price Versus Year of Built", style={'textAlign': 'center', 'padding-top': 5}),
            dcc.Graph(
                id='price vs built_year',
                figure = fig_row6
            ),

    html.H3("Square Feet Versus Price", style={'textAlign': 'center', 'padding-top': 5}),
            dcc.Graph(
                id='Sqft vs Price 1',
                figure = fig_row7a
            )

        ]),

        ], className="container"),

    dcc.Tab(label = 'Housing Attributes Relationship',
            style={"textAlign": "center",
                   "color": "#800080",
                    "font-family": 'Open Sans', "font-size": '3rem',
                    "font-weight": "800", "text-transform:": "uppercase",
                    "line-height": "normal",
                    },
            children=[
    html.H3("Proportion of each attribute", style={'textAlign': 'center', 'padding-top': 5}),
            html.Div([
                dcc.Dropdown(
                    id='pie_dropdown',
                    options=[{'label': i, 'value': i} for i in pie_list],
                    value=pie_list[3]
                ),
            ], style={'width': '50%', "align": "center", "justify-content": "center", 'display': 'inline-block'}),
            dcc.Graph(id='pie'),

    html.H3("Distribution of each attribute", style={'textAlign': 'center', 'padding-top': 5}),
            html.Div([
                dcc.Dropdown(
                    id='type_dropdown',
                    options=[{'label': i, 'value': i} for i in type_list],
                    value=type_list[0])
            ], style={'width': '50%', "align": "center", "justify-content": "center", 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                    id='bar_dropdown',
                    options=[{'label': i, 'value': i} for i in columns],
                    value=columns[3])
        ], style={'width': '50%', "align": "center", "justify-content": "center", 'display': 'inline-block'}),
        dcc.Graph(id='bar'),

    html.H3("Room Type/ Unit Type Report", style={'textAlign': 'center', 'padding-top': 5}),
            dcc.Graph(
                id='room type graph',
                figure={
                    'data': [
                    {'x': data1.index,
                    'marker': {'color': ['#4CC417',' #52D017', '#4CC552',
                                         '#54C571', '#7FE817', '#59E817',
                                         '#57E964', '#64E986', '#64E986',
                                ]},
                     'y': data1, 'type': 'bar'},
                    ],
                    'layout': {
                    'xaxis': {
                        'type': 'category',
                        'title': 'Room/ Unit type',
                    },
                    'yaxis': {
                        'range': [0, 4000],
                        'title': 'Quantity'
                    },
                    'title': 'Most Popular Room Type in Malaysia',
                    }
                }
            ),

    html.H3("HeatMap Correlation Matrix", style={'textAlign': 'center', 'padding-top': 5}),
            dcc.Graph(id = "heatmap",
                  figure={
                      'data':[
                          go.Heatmap(z = np.fabs(corr.values),
                              x = corr.columns.tolist(),
                              y = corr.index.tolist(),
                              colorscale='YlGnBu')],
                      'layout':go.Layout(margin=dict(l=100, b=100, t=50))
                  }
            ),

    html.H3("Pearson Correlation of Housing Attributes", style={'textAlign': 'center', 'padding-top': 5}),
        html.H6("Brief Housing Data Analysis", style={'textAlign': 'center', 'padding-top': 5}),
        dcc.Markdown('''     
            + From Pearson Correlation table, housing price is highly correlated with Number of Bathroom and Bedroom
            + The year of built, house area (square feet) and other attributes are evenly distributed,
            so less correlated each other.
            +  Terraced House occupies 50.67% which is almost half of percentage of Malaysia House Categories.
            +  In average, most of the house has 4 bedroom and 3 bathroom, which is also the most popular type.
            +  Most of the house is unfurnished and located at ground floor.
        '''),

        html.Br([]),
        html.H3("Wordcloud Picture", style={'textAlign': 'center', 'padding-top': 5}),
        html.Img(id='image_wc',
                    style={
                     'height': '200%',
                     'width': '100%'
                    }
                 ),

    ], className="container"),

    dcc.Tab(label='Map Visualization', style={"textAlign": "center", "color": "#008000",
                                            "font-family": 'Open Sans', "font-size": '3rem',
                                            "font-weight": "800", "text-transform:": "uppercase",
                                             "line-height": "normal",
                                             },
            children=[
        html.H3("Selection Panel", style={'textAlign': 'center', 'padding-top': 5}),
        html.H5("1. Select a State"),
            html.Div(
                className="div-for-dropdown",
                children=[
                    dcc.Dropdown(id='state-dropdown',
                                 options=[{'label': i, 'value': i} for i in df['State'].unique()],
                                 value='Select a State'
                                 ),
                ], style={'border-bottom': 'solid 3px', 'border-color':'#000000','padding-top': '6px'}
            ),
            html.Br([]),
        html.H5("2. Select a Category"),
            html.Div(
                className="div-for-dropdown",
                children=[
                    dcc.Checklist(id='category-dropdown',
                                options = [{'label':str(b),'value':b} for b in sorted(df['Category'].unique())],
                                  value = [b for b in sorted(df['Category'].unique())],
                                  ),
                ],
            ),
            html.Div(
                html.Ul([
                    html.Li("Apartment/Flat", className='circle',
                            style={'background': '#ff00ff', 'color': 'black',
                                    'list-style': 'none', 'text-indent': '17px'}),
                    html.Li("Bungalow", className='circle',
                            style={'background': '#0000ff', 'color': 'black',
                                    'list-style': 'none', 'text-indent': '17px',
                                    'white-space': 'nowrap'}),
                    html.Li("Condo", className='circle',
                            style={'background': '#FF0000', 'color': 'black',
                                    'list-style': 'none', 'text-indent': '17px'}),
                    html.Li("Penthouse/Duplex", className='circle',
                            style={'background': '#00ff00', 'color': 'black',
                                    'list-style': 'none', 'text-indent': '17px'}),
                    html.Li("Semi-D", className='circle',
                            style={'background': '#824100', 'color': 'black',
                                    'list-style': 'none', 'text-indent': '17px'}),
                    html.Li("Terraced House", className='circle',
                            style={'background': '#DFFF00', 'color': 'black',
                                    'list-style': 'none', 'text-indent': '17px'}),
                    html.Li("Townhouse", className='circle',
                            style={'background': '#CCCCFF', 'color': 'black',
                                    'list-style': 'none', 'text-indent': '17px'}),

                ]), style={'border-bottom': 'solid 3px', 'border-color':'#000000','padding-top': '6px'}

            ),
            html.Br([]),
        html.H5("3. Select a Price Range"),
            html.Div(
                className="div-for-dropdown",
                children=[
                dcc.RangeSlider(
                id='price-range-slider',
                min = df["Price"].min(),
                max = df["Price"].max(),
                marks = {
                    45000   : {'label': 'RM45,000',    'style': {'color': '#FF0000'}},
                    300000  : {'label': 'RM300,000',   'style': {'color': '#CD5C5C'}},
                    550000  : {'label': 'RM550,000',   'style': {'color': '#008000'}},
                    800000  : {'label': 'RM800,000',   'style': {'color': '#000080'}},
                    1100000 : {'label': 'RM1,100,000', 'style': {'color': '#0000FF'}},
                    1400000 : {'label': 'RM1,400,000', 'style': {'color': '#FF00FF'}},
                    1700000 : {'label': 'RM1,700,000', 'style': {'color': '#7302A6'}},
                    2000000 : {'label': 'RM2,000,000', 'style': {'color': '#800080'}},
                },
                value = [df["Price"].min(), df["Price"].min(),]
                ),
                html.Div(id='output-price-range-slider')
                ],#close of children
            ),
            html.Br([]),
        html.H3("House Distribution on Map View", style={'textAlign': 'center', 'padding-top': 5}),
            html.Div(
                className = "row",
                id="top-row-graphs",
                children = [
                    dcc.RadioItems(
                        id="mapbox-view-selector",
                        options=[
                            {"label": "Basic", "value": "basic"},
                            {"label": "Streets", "value": "streets"},
                            {"label": "Satellite", "value": "satellite"},
                            {"label": "Outdoors", "value": "outdoors"},
                            {
                                "label": "Satellite-street",
                                "value": "mapbox://styles/mapbox/satellite-streets-v9",
                            },
                        ], labelStyle={'display': 'inline-block'},
                        value="basic",
                    ),
                    dcc.Graph(
                        id="map-graph",
                        config={"displayModeBar": False, "scrollZoom": True},
                        style={'background':'#00FC87', 'height':'100vh'}
                    ),
                ]
            )
    ], className="container"),

    dcc.Tab(label='House Price Prediction', style={"textAlign": "center", "color": "#FF00FF",
                                            "font-family": 'Open Sans', "font-size": '3rem',
                                            "font-weight": "800", "text-transform:": "uppercase",
                                             "line-height": "normal",
                                             },
            children=[
        html.H2("1.0 System View", style={'textAlign': 'center', 'padding-top': 5}),
        html.H3("1.1 Actual VS Predicted Housing Price - Table", style={'textAlign': 'center', 'padding-top': 5}),
            dcc.Graph(
                id="avp-graph",
                figure=avp_fig,
                style={"height": "450px"}
            ),
        html.H3("1.2 Actual VS Predicted Housing Price - Table", style={'textAlign': 'center', 'padding-top': 5}),
        dash_table.DataTable(
            id='datatable-interactivity',
            columns=[
                {"name": i, "id": i, "deletable": True, "selectable": True}
                for i in prediction_result.columns
            ],
            data = prediction_result.to_dict('records'),
            export_format="csv",
            editable=True,
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            column_selectable="single",
            row_selectable="multi",
            selected_columns=[],
            selected_rows=[],
            page_action="native",
            page_current=0,
            page_size=10,
            style_cell={'textAlign': 'left',
                        'backgroundColor': '#92a8d1',
                        'color': 'white'
                        },
            style_header={
                'border': '1px solid blue',
                'backgroundColor': 'Black',
                'fontWeight': 'bold'
            },
            style_cell_conditional=[
                    {
                        'if': {'column_id': 'Type'},
                        'textAlign': 'left'
                    }
            ],
            style_data={
                'whiteSpace': 'normal',
                'height': 'auto',
                'lineHeight': '15px',
                'border': '1px solid pink'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ],
        ),

        html.H3("1.3 Actual VS Predicted Housing Price - ScatterGraph", style={'textAlign': 'center', 'padding-top': 5}),
        html.Div(
            className="predict-dropdown-type",
            children=[
                dcc.Dropdown(id='predict-type-dropdown', options=[
                    {'label': '1-storey Terraced House', 'value': '1-storey Terraced House'},
                    {'label': '1.5-storey Terraced House', 'value': '1.5-storey Terraced House'},
                    {'label': '2-storey Terraced House', 'value': '2-storey Terraced House'},
                    {'label': '2.5-storey Terraced House', 'value': '2.5-storey Terraced House'},
                    {'label': '3-storey Terraced House', 'value': '3-storey Terraced House'},
                    {'label': '3.5-storey Terraced House', 'value': '3.5-storey Terraced House'},
                    {'label': 'Apartment', 'value': 'Apartment'},
                    {'label': 'Bungalow', 'value': 'Bungalow'},
                    {'label': 'Cluster House', 'value': 'Cluster House'},
                    {'label': 'Condominium', 'value': 'Condominium'},
                    {'label': 'Link Bungalow', 'value': 'Link Bungalow'},
                    {'label': 'Penthouse', 'value': 'Penthouse'},
                    {'label': 'Semi-Detached House', 'value': 'Semi-Detached House'},
                    {'label': 'Service Residence', 'value': 'Service Residence'},
                    {'label': 'Studio', 'value': 'Studio'},
                    {'label': 'Terraced House', 'value': 'Terraced House'},
                    {'label': 'Townhouse', 'value': 'Townhouse'},
                    {'label': 'Townhouse Condo', 'value': 'Townhouse Condo'},
                    {'label': 'Twin Courtyard Villa', 'value': 'Twin Courtyard Villa'},
                    {'label': 'Twin Villas', 'value': 'Twin Villas'},
                    {'label': 'Zero-Lot Bungalow', 'value': 'Zero-Lot Bungalow'},
                ], multi=True, value=['Apartment'],
                             ),
            ], style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "60%"}
        ),
        dcc.Graph(id="predict-line-chart"),

        html.Br([]),
        html.H2("2.0 User View", style={'textAlign': 'center', 'padding-top': 5}),
        html.H3("2.1 Upload Your Dataset:", style={'textAlign': 'center', 'padding-top': 5}),
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            # Allow multiple files to be uploaded
            multiple = True,
        ),
        html.Div(id="output-data-upload"),
    ], className="container")
    ])
]) #end layout

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    df_upload_prediction = df
    cols = ['Bedroom', 'Bathroom', 'Square_feet', 'Price per Sqft']
    upload_x = df_upload_prediction[cols].values
    upload_y = df_upload_prediction['Price'].values

    upload_X_train, upload_X_test, upload_y_train, upload_y_test = train_test_split(
        upload_x, upload_y, test_size=0.33, random_state=42
    )

    clf = RandomForestRegressor(n_estimators=400)
    clf.fit(upload_X_train, upload_y_train)
    upload_y_pred = clf.predict(upload_X_test)

    rfr = clf
    upload_testing_data = df_upload_prediction[cols]
    upload_final_test = upload_testing_data.values
    upload_y_te_pred = rfr.predict(upload_final_test)
    upload_prediction = pd.DataFrame(upload_y_te_pred, columns=['Predicted Price'])
    upload_prediction_result = pd.concat([df_upload_prediction['ID'],
                                          df_upload_prediction['Type'],
                                          df_upload_prediction['Bedroom'],
                                          df_upload_prediction['Bathroom'],
                                          df_upload_prediction['Square_feet'],
                                          df_upload_prediction['Price per Sqft'],
                                          df_upload_prediction['Price'], prediction],
                                         axis=1)
    upload_prediction_result = upload_prediction_result.astype({"Predicted Price": int})
    upload_test_mae = metrics.mean_absolute_error(upload_y_test, upload_y_pred)

    user_upload_fig = px.scatter(
        x = y_test,
        y = y_pred,
        labels={"x": "Actual", "y": "Predicted"},
        title=f"Actual vs predicted interest rate (MAE={upload_test_mae:.2f})",
    )
    user_upload_fig.add_shape(
        type = "line",
        x0 = y_test.min(),
        y0 = y_test.min(),
        x1 = y_test.max(),
        y1 = y_test.max()
    )

    #return table and graph
    return html.Div([
        html.H4("2.2 Actual VS Predicted Housing Price - LineGraph", style={'textAlign': 'center', 'padding-top': 5}),
            dcc.Graph(
                id = "user-upload-graph",
                figure = user_upload_fig,
                style={"height": "450px"}
            ),
        html.H4("2.3 Table Uploaded by User", style={'textAlign': 'center', 'padding-top': 5}),
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),
        dash_table.DataTable(
            data = df.to_dict('records'),
            columns = [{'name': i, 'id': i} for i in df.columns],
            editable=True,
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            column_selectable="single",
            row_selectable="multi",
            selected_columns=[],
            selected_rows=[],
            page_action="native",
            page_current=0,
            page_size=10,
            style_cell={'textAlign': 'left',
                        'backgroundColor': '#92a8d1',
                        'color': 'white'
                        },
            style_header={
                'border': '1px solid blue',
                'backgroundColor': 'Black',
                'fontWeight': 'bold'
            },
            style_cell_conditional=[
                {
                    'if': {'column_id': 'Type'},
                    'textAlign': 'left'
                }
            ],
            style_data={
                'whiteSpace': 'normal',
                'height': 'auto',
                'lineHeight': '15px',
                'border': '1px solid pink'
            },
        ),

        html.H4("2.4 New Table Predicted by System", style={'textAlign': 'center', 'padding-top': 5}),
        dash_table.DataTable(
            id='upload-datatable-system',
            columns=[
                {"name": i, "id": i, "deletable": True, "selectable": True}
                for i in upload_prediction_result.columns
            ],
            data = upload_prediction_result.to_dict('records'),
            export_format = "csv",
            editable=True,
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            column_selectable="single",
            row_selectable="multi",
            selected_columns=[],
            selected_rows=[],
            page_action="native",
            page_current=0,
            page_size=10,
            style_cell={'textAlign': 'left',
                        'backgroundColor': '#92a8d1',
                        'color': 'white'
                        },
            style_header={
                'border': '1px solid blue',
                'backgroundColor': 'Black',
                'fontWeight': 'bold'
            },
            style_cell_conditional=[
                {
                    'if': {'column_id': 'Type'},
                    'textAlign': 'left'
                }
            ],
            style_data={
                'whiteSpace': 'normal',
                'height': 'auto',
                'lineHeight': '15px',
                'border': '1px solid pink'
            },
        ),
        html.Hr(),  # horizontal line
        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])#end of return

@app.callback(
    dash.dependencies.Output("output-data-upload", "children"),
    [dash.dependencies.Input("upload-data", "contents")],
    [dash.dependencies.State("upload-data", "filename"),
    dash.dependencies.State("upload-data", "last_modified")]
)
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

#row7
@app.callback(dash.dependencies.Output('image_wc', 'src'),
              [dash.dependencies.Input('image_wc', 'id')])
def make_image(b):
    img = BytesIO()
    plot_wordcloud().save(img, format='PNG')
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())

#row3
@app.callback(
    dash.dependencies.Output('pie', 'figure'),
    [dash.dependencies.Input('pie_dropdown', 'value')
     ])
def update_pie(value):
    count = df_count.groupby(value).count()
    trace = go.Pie(labels = count['计数'].index.tolist(), values = count['计数'].tolist())
    layout = go.Layout(margin = dict(t = 0, b = 0), height = 400)
    return dict(data=[trace], layout=layout)

#row3
@app.callback(
dash.dependencies.Output('bar', 'figure'),
[dash.dependencies.Input('bar_dropdown', 'value'),
 dash.dependencies.Input('pie_dropdown', 'value'),
 dash.dependencies.Input('type_dropdown', 'value')
])
def update_bar(value0, value1, type):
    cross = pd.crosstab(dfi[value0], dfi[value1], margins=True)
    cross_col_name = cross.columns.tolist()[:-1]
    cross_ = cross.copy()
    for name in cross_col_name:
        cross_[name] = cross_[name] / cross_['All']
    if type == 'Number':
        cross_new = cross.iloc[:-1, :-1]
    else:
        cross_new = cross_.iloc[:-1, :-1]
    data = []
    for key in cross_new.columns.tolist():
        trace = go.Bar(
            x=cross_new.index.tolist(),
            y=cross_new[key].tolist(),
            name=key,
            opacity=0.6
        )
        data.append(trace)
    layout = go.Layout(barmode='stack', margin=dict(t=0, b=30),height=400)
    fig = go.Figure(data=data, layout=layout)
    return fig

#row10
@app.callback(
    dash.dependencies.Output('output-price-range-slider', 'children'),
    [dash.dependencies.Input('price-range-slider', 'value')])
def update_output(value):
    return 'You have selected price range between RM {}'.format(value)

#state-dropdown
#price-range-slider
#category-dropdown
#output of map graph
@app.callback(
    dash.dependencies.Output("map-graph", "figure"),
    [
        dash.dependencies.Input("state-dropdown", "value"),
        dash.dependencies.Input("category-dropdown", "value"),
        dash.dependencies.Input("price-range-slider", "value"),
        dash.dependencies.Input("mapbox-view-selector", "value"),
    ],
)
def update_figure(chosen_state, chosen_category, chosen_price, mapbox_view):
    #df_sub = df[(df['State'].isin(chosen_state)) &
                #(df['Category'].isin(chosen_category))
                #(df['Price'] > price[chosen_price(0)])
                #(df['Price'] < price[chosen_price(1)])
                #]
    df_sub = df.loc[( df.State == chosen_state ) &
                    ( df['Category'].isin(chosen_category) ) &
                    ( df['Price'].between(chosen_price[0], chosen_price[1], inclusive = True) )
                    ]
    my_text=['Name</b>:' + name +'<br><b>Price</b>: RM' + str(price) + '<br><b>Type</b>: ' + houseType
             +'<br><b>Ownership</b>: '+ Ownership + '<br><b>Status</b>: ' + Furnishing
    for name, price, houseType, Ownership, Furnishing
             in zip(list(df_sub['Name']),
                    list(df_sub['Price']),
                    list(df_sub['Type']),
                    list(df_sub['Ownership']),
                    list(df_sub['Furnishing'])
                    )
            ]
    # Create figure
    locations = [go.Scattermapbox(
        lat = df_sub['Latitude'],
        lon = df_sub['Longitude'],
        mode='markers',
        marker = {'color' : df_sub['color']},
        unselected = {'marker' : {'opacity':1}},
        selected = {'marker' : {'opacity':0.5, 'size':25}},
        text = my_text
    )]
    # Return figure
    return {
        'data': locations,
        'layout': go.Layout(
            uirevision= 'foo', #preserves state of figure/map after callback activated
            clickmode = 'event+select',
            #showlegend=True,
            autosize=True,
            hovermode = 'closest',
            hoverdistance = 2,
            mapbox=dict(
                accesstoken = mapbox_access_token,
                bearing = 25,
                style = mapbox_view,
                center = go.layout.mapbox.Center(
                    lat = 3.140853,
                    lon = 101.693207
                ),
                pitch = 40,
                zoom = 5
            ),
            margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
        )
    }

#predicted-result-table
@app.callback(
    dash.dependencies.Output('datatable-interactivity', 'style_data_conditional'),
    [dash.dependencies.Input('datatable-interactivity', 'selected_columns')]
)
def update_styles(selected_columns):
    return [{
        'if': { 'column_id': i },
        'background_color': '#D2F3FF'
    } for i in selected_columns]

#row12
@app.callback(
    dash.dependencies.Output('predict-line-chart', 'figure'),
    [dash.dependencies.Input('predict-type-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"1-storey Terraced House":  "1-storey Terraced House",
                "1.5-storey Terraced House": "1.5-storey Terraced House",
                "2-storey Terraced House":  "2-storey Terraced House",
                "2.5-storey Terraced House": "2.5-storey Terraced House",
                "3-storey Terraced House":  "3-storey Terraced House",
                "3.5-storey Terraced House":  "3.5-storey Terraced House",
                "Apartment":  "Apartment",
                "Bungalow":  "Bungalow",
                "Cluster House":  "Cluster House",
                "Condominium":  "Condominium",
                "Link Bungalow":  "Link Bungalow",
                "Penthouse":  "Penthouse",
                "Semi-Detached House":  "Semi-Detached House",
                "Service Residence":  "Service Residence",
                "Studio":  "Studio",
                "Terraced House":  "Terraced House",
                "Townhouse":  "Townhouse",
                "Townhouse Condo":  "Townhouse Condo",
                "Twin Courtyard Villa":  "Twin Courtyard Villa",
                "Twin Villas":  "Twin Villas",
                "Zero-Lot Bungalow":  "Zero-Lot Bungalow",}
    trace1 = []
    trace2 = []
    for house_type in selected_dropdown:
        trace1.append(
          go.Scatter(x = prediction_result[prediction_result["Type"] == house_type]["Square_feet"],
                     y = prediction_result[prediction_result["Type"] == house_type]["Price"],
                     mode = 'lines', opacity=0.7,
                     name = f'Actual Price of {dropdown[house_type]}',textposition='bottom center'))
        trace2.append(
          go.Scatter(x = prediction_result[prediction_result["Type"] == house_type]["Square_feet"],
                     y = prediction_result[prediction_result["Type"] == house_type]["Predicted Price"],
                     mode = 'lines', opacity=0.6,
                     name = f'Predicted Price of {dropdown[house_type]}',textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1',
                                            '#FF7400', '#FFF400', '#FF0056'],
            height = 600,
            title = f"Actual and Predicted Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
             xaxis = {"title": "Square_feet",
                      'rangeslider': {'visible': True}},
             yaxis = {"title":"Price (MYR)"})
              }
    return figure

if __name__ == '__main__':
    app.run_server(debug=True)