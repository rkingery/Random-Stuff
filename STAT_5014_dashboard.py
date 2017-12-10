import numpy as np
import plotly
import plotly.graph_objs as go
np.random.seed(42)

iters = 5000
ps = np.arange(50-3,50+3+1)*.01
t = range(1,iters+1)
traces = []
button_list = []

for p in ps:
    x = np.random.choice([1,-1],size=iters,p=[p,1-p])
    s = np.cumsum(x)
    trace = go.Scatter(
            x = t,
            y = s,
            name = 'p = '+str(p),
            line = dict(width = 2)
    )
    traces += [trace]
    button = dict(label = 'p = '+str(p),
              method = 'update',
              args = [{'visible': list(ps == p)}])
                      #{'title': 'Random walk with p = '+str(p)} ])
    button_list += [button]
    
data = traces

updatemenus = list([
    dict(type="buttons",
         active=-1,
         buttons=button_list)
])


layout = dict(title = 'Fun with Random Walks',
              xaxis = dict(title = 'iterations'),
              yaxis = dict(title = 'position'),
              showlegend=False,
              updatemenus=updatemenus)

fig = dict(data=data, layout=layout)
plotly.offline.plot(fig)