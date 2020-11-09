# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import base64
import datetime
import io
import os

import json

import dash
import dash_auth
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import dash_bootstrap_components as dbc

import flask
from flask_caching import Cache

import plotly.express as px
import pandas as pd

import dash_reusables as dr

import evalml
from evalml import AutoMLSearch

import uuid

import db_queries as db

VALID_USERNAME_PASSWORD_PAIRS = db.get_users()


image_directory = os.getcwd()
static_image_route = '/static/'


pages = ['load-data-link','view-data-link','automl-link']


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MATERIA])

app.config.suppress_callback_exceptions = True

auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)




# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 100,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    #"background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    #"margin-left": "18rem",
    #"margin-right": "2rem",
    "padding": "2rem 1rem",
}



cache = Cache(app.server, config={
    #'CACHE_TYPE': 'redis',
    # Note that filesystem cache doesn't work on systems with ephemeral
    # filesystems like Heroku.
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory',

    # should be equal to maximum number of users on the app at a single time
    # higher numbers will store more data in the filesystem / redis cache
    'CACHE_THRESHOLD': 10
})


def get_dataframe(session_id, filename = None, sep = '|', index_col = None):
    @cache.memoize()
    def query_and_serialize_data(session_id):

        id_col = int(index_col) if index_col else index_col
        print(filename)

        
        if 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(filename, index_col=id_col)
        else:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(filename, sep = sep, index_col = id_col)
            #print(df.head())
            
        return df.to_json()

    return pd.read_json(query_and_serialize_data(session_id))




data_div = html.Div([
                dbc.Row(
                         [
                             dbc.Col(
                             [
                              html.Div(
                                          [
                                              html.H3('Start by telling us about your data...'),
                                              html.Br(),
                                              html.Label("Data Separator: "),
                                              dcc.Dropdown( id='data_sep_dropdown',
                                                            options=[{'label':'pipe', 'value': '|'},
                                                                     {'label':'comma','value': ','},
                                                                     {'label':'tab','value': '\t'}] ,
                                                            value='pipe',
                                                            multi=False,
                                                           #style={'float': 'right','margin': 'auto'}
                                                          ),
                                              html.Label('Index Column: '),
                                              dcc.Input( id='data_index_input',
                                                         value=None,
                                                         placeholder='Enter Index Column if present',
                                                         style={'width': '99%'}
                                                       )
                                          ]
                                         ,
                                        id='div_file_info'
                                #style={'columnCount': 2}
                              ),

                                dcc.Upload(
                                    id='upload-data',
                                    children=html.Div([
                                        'Drag and Drop or ',
                                        html.A('Select Files')
                                    ]),
                                    style={
                                        'width': '99%',
                                        'height': '60px',
                                        'lineHeight': '60px',
                                        'borderWidth': '1px',
                                        'borderStyle': 'dashed',
                                        'borderRadius': '5px',
                                        'textAlign': 'center',
                                        'margin': '10px'
                                    },
                                    # Allow multiple files to be uploaded
                                    multiple=False
                                ),

                                html.Div(id='output-data-upload'),

                                #html.Label('Load your data file'),
                                #html.Div(
                                #         ["filename: ",dcc.Input(id='data_filename', value='', type='text')],
                                #            id='div_file_input'),
                                #html.Button(id='submit-button-state', n_clicks=0, children='Submit'),

                                html.Br(),


                                html.Label('Features to include'),
                                html.Div(dcc.Dropdown(id='feature_dropdown'), id='feature-dropdown-div'),
                                html.Br(),
                                html.Label('Target'),
                                dcc.Dropdown(id='target_dropdown',persistence=True),
                                html.Br()
                             ],
                                 width = 5
                         ),

                             dbc.Col(html.Div(id='data_table'))
                             
                                     
                         ]
                )
]
)

automl_row1 = dbc.Row(
    [
        dbc.Col(
                [
                    html.H3("We're ready to start modeling..."),
                    html.Br(),
                    html.Label('How many models do you want to train?'),
                    dcc.Slider(id='num-models-slider',
                                min=0,
                                max=100,
                                marks={i: str(i) for i in range(0, 105, 20)},
                                value=5,
                            ), 
                    html.Br(),
                    dbc.Button('Start AutoML',color="primary",id='start_automl_button', n_clicks=0)
                ]
        ),
        dbc.Col(html.Div(dbc.Spinner(html.Div(id='ml-table-div'))))
    ]
)


automl_row2 = dbc.Row( 
    [
        dbc.Col([html.Div(id='roc-fig'),
                html.Div(id='conf-matrix-fig'),
                html.Div(id='obj-thresh-fig')
                ]),
        dbc.Col(html.Div(id='feature-imp-fig'))
    ]
)

automl_cards = html.Div(id='automl-div',
                        children = [
                            automl_row1,
                            automl_row2,
                            dbc.Row(dbc.Col(html.Div(id='ml-pipeline-fig-div',
                                                              style={'textAlign': 'center'}), 
                                           )
                                    )
                                   ]
                       )
#automl_div = html.Div(id='automl-div',
#                     children=
#                        html.Br(),
#                        html.Br(),
#                        dbc.Spinner(html.Div(id='ml-table-div')),
#                        html.Div(id='ml-pipeline-fig-div',style={'textAlign': 'center'})
#             ]
#            )


navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Load Data", href="/load-data", id="load-data-link")),
        dbc.NavItem(dbc.NavLink("View Data", href="/view-data", id="view-data-link")),
        dbc.NavItem(dbc.NavLink("AutoML", href="/automl", id="automl-link")),
    ],
    brand="EvalML Made Simple",
    brand_href="#",
    color="primary",
    dark=True,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)



def serve_layout():
    session_id = str(uuid.uuid4())
    print(session_id)

    return html.Div([
        dcc.Location(id="url", refresh=False), 
        
        # Hidden divs inside the app that stores the intermediate value
        dcc.Store(id='cached-features', storage_type='session'),
        dcc.Store(id='cached-target', storage_type='session'),
        dcc.Store(id='cached-best-pipeline',storage_type='session'),
        html.Div(session_id, id='session-id', style={'display': 'none'}),
        navbar, 
        #html.H1(children='EvalML Automated Machine Learning Platform'),
        #html.H2(children='''Open Source AutoML made easy.'''),
        html.Br(),
        html.Br(),        
        content
    ])


app.layout = serve_layout


# this callback uses the current pathname to set the active state of the
# corresponding nav link to true, allowing users to tell see page they are on
@app.callback(
    [Output(f"{p}-link", "active") for p in pages],
    [Input("url", "pathname")],
)
def toggle_active_links(pathname):
    if pathname == "/":
        # Treat page 1 as the homepage / index
        return True, False, False
    return [pathname == f"/{c[:-5]}" for c in pages]


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname in ["/", "/load-data"]:
        return data_div#html.P("This is the content of page 1!")
    elif pathname == "/view-data":
        return html.P("This is the content of page 2. Yay!")
    elif pathname == "/automl":
        return automl_cards#html.P("Oh cool, this is page 3!")
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )

########### Reading in the dataframe ###############

def parse_contents(contents, data_sep, index_col, filename, date, session_id):
    try:
        get_dataframe(session_id, filename, sep=data_sep, index_col = index_col)
        return f'Successfully loaded {filename}'
        
    except Exception as e:
        print(e)
        return f'Failed to load {filename}'
        #return html.Div([
        #    'There was an error processing this file.'
        #])

@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State(component_id='data_sep_dropdown', component_property='value'),
               State(component_id='data_index_input', component_property='value'),
               State('session-id','children'),
               State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(contents, data_sep, index_col, session_id, name, date):
    if contents is not None:
        return parse_contents(contents, data_sep, index_col, name, date, session_id)


    
@app.callback(
    [Output('feature-dropdown-div','children'),
     Output('cached-features','data')],
    [Input(component_id='output-data-upload', component_property='children'),
     Input('session-id', 'children')]
)
def update_features_dropdown(input_value,session_id):
    df = get_dataframe(session_id)
    print(df.head())
    return  ( dcc.Dropdown(id='feature_dropdown',
                         options=[{'label': i, 'value': i} for i in df.columns], 
                         value = df.columns.tolist(),
                         multi=True,
                         persistence=True),
             {'features':df.columns.tolist()})
    
@app.callback(
    [Output(component_id='data_table', component_property='children'),
     Output(component_id='target_dropdown', component_property='options')],
    [Input(component_id='feature_dropdown', component_property='value'),
     Input('session-id', 'children')]
)
def update_features_div(input_value,session_id):
    df = get_dataframe(session_id)[input_value]
    return (dash_table.DataTable(
                                data=df.to_dict('records'),
                                columns=[{'id': c, 'name': c} for c in input_value],
                                page_size=20,
                                style_table={'overflowY': 'auto'}), 
            [{'label': i, 'value': i} for i in input_value])
    #dr.generate_table(df[input_value]), [{'label': i, 'value': i} for i in input_value]


@app.callback(Output('cached-target','data'),
              Input('target_dropdown','value'))
def update_cached_target(value):
    return value
    
    
@app.callback([Output('ml-table-div','children'),
               Output('ml-pipeline-fig-div','children'),
               Output('roc-fig','children'),
               Output('feature-imp-fig','children'),
               Output('conf-matrix-fig','children'),
               Output('obj-thresh-fig','children'),
               Output('cached-best-pipeline','data')],
              [Input('start_automl_button','n_clicks')],
              [State('session-id', 'children'),
               State('cached-features', 'data'),
               State('cached-target', 'data'),
               State('num-models-slider','value')])
def run_AutoML(clicks, session_id, features, target, num_models):
    if not clicks:
        return '',''
        
    #print(clicks)
    #print(features)
    #print(target)
    
    df = get_dataframe(session_id)
    print(df.dtypes)
    features = features['features']
    target = target
    
    
    print(len(df))
    
    if len(df)==0:
        return '',''

    X = df[features].drop(target,axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = evalml.preprocessing.split_data(X, y, test_size=.3)
    
    automl = AutoMLSearch(problem_type="binary", objective="f1", max_iterations=num_models)
    
    automl.search(X_train, y_train)
    
    results_df = automl.rankings[automl.rankings.columns.tolist()[:-1]]
    
    pipeline = automl.best_pipeline
    pipeline.fit(X_train, y_train)
    #pipeline.score(X_test, y_test, ["auc"])
    y_pred = pipeline.predict(X_test)
    
    best_graph = automl.best_pipeline.graph()
    best_graph.format = 'png'
    image_filename = 'best_pipeline_image'
    best_graph.render(image_filename)
    encoded_image = base64.b64encode(open(os.getcwd()+'/'+image_filename+'.png', 'rb').read())
    
    return [[html.H3('Results:'),
             html.Br(),
             dash_table.DataTable(
                                data=results_df.to_dict('records'),
                                columns=[{'id': c, 'name': c} for c in results_df.columns],
                                page_action='none',
                                style_table={'height': '300px', 'overflowY': 'auto'})
            ],
            html.Img(src=static_image_route + image_filename + '.png'),
            dcc.Graph(figure=evalml.model_understanding.graph_roc_curve(y_test,pipeline.predict_proba(X_test))),
            dcc.Graph(figure=pipeline.graph_feature_importance()),
            dcc.Graph(figure=evalml.model_understanding.graph_confusion_matrix(y_test,y_pred)),
            dcc.Graph(figure=evalml.model_understanding.graph_binary_objective_vs_threshold(pipeline,
                                                                           X_test,
                                                                           y_test,
                                                                           'f1',
                                                                           steps=100)
                     ),
            automl.rankings.loc[:,'parameters'].to_dict()           
           ]

#'data:image/png;base64,{}'.format(encoded_image)))#os.getcwd()+'/'+image_filename+'.png'))#
    #return dr.generate_table(results_df)


# Add a static image route that serves images from desktop
# Be *very* careful here - you don't want to serve arbitrary files
# from your computer or server
@app.server.route('{}<image_path>.png'.format(static_image_route))
def serve_image(image_path):
    image_name = '{}.png'.format(image_path)
    #if image_name not in list_of_images:
    #    raise Exception('"{}" is excluded from the allowed static files'.format(image_path))
    return flask.send_from_directory(image_directory, image_name)        

if __name__ == '__main__':
    app.run_server(debug=True)