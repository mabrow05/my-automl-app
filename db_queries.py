import sqlite3 as sql
import pandas as pd
import json

# start connection to database
conn = sql.connect('automl.db')
c = conn.cursor()

project_columns = ['user_id', 'username', 'name', 'prediction_type', 'features', 'features_selected', 'target']


def get_users():
    
    c.execute('SELECT * FROM users')
    return {u[1]:u[2] for u in c.fetchall()}

def get_projects(username):
    return pd.read_sql('SELECT * FROM project where username = {}'.format(username), conn)

def get_data(username,projectname):
    return pd.read_sql('SELECT * FROM {}'.format(username+'_'+projectname), conn)

def write_project(projectname, username, userid, prediction_type, features, features_used, target):
    
    data = {
            project_columns[0]:[userid],
            project_columns[1]:[username],
            project_columns[2]:[projectname],
            project_columns[3]:[prediction_type],
            project_columns[4]:[json.dumps(features)],
            project_columns[5]:[json.dumps(features_used)],
            project_columns[6]:[target]
           }
    
    with sql.connect('automl.db') as conn: 
        pd.DataFrame(data).to_sql('project',conn, if_exists='append')
        
def write_data(username, projectname, df):
    
    with sql.connect('automl.db') as conn:
        df.to_sql(usernamename+'_'+projectname, conn, if_exists='overwrite')
        
def write_model_results(projectname, projectid, results):
    pass
    
 