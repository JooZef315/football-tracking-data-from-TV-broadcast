import pandas as pd

def get_positions(df, colors):  
  '''
  this functions is to decide if the boject is in Attacking team or Defense team, or if it is a ball.

  Parameters
  ----------
  df : dataframe
    the dataframe of the objects.
  colors : list of strings         
    list of the colors to classify the teams.    
  
  Return
  ----------
  df : dataframe
      the dataframe of the objects and their positions. 
  '''

  positions = []
  for i, c in enumerate(df['Color'].tolist()):    
    if df['Class'][i] == 'Ball':
      positions.append('Ball')
      continue
    
    if df['Color'][i] == colors[0] or df['Color'][i] == colors[1]: 
      positions.append('Defense')
    elif df['Color'][i] == colors[2] or df['Color'][i] == colors[3]:
      positions.append('Attack')
    else:
      positions.append('Not a player')

  df['Position'] = positions
  return df