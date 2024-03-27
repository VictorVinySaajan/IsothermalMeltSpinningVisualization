import dash
from dash import Dash
from dash import html
import dash_daq as daq
from dash import dcc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from numpy import linalg as LA
import math
import dash_latex as dl
from ml_utils import MLUtils
from vis_utils import VisUtils

ml_util_object = MLUtils()
vis_utils_object = VisUtils()
# ml_util_object.getPredictions(np.array([[1, 2, 3, 4, 5, 6, 7]]))

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
mathjax = ['https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML']
app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server

analysis_text_global = 'Click for Bivariate Analysis-->'
space_text_global = 'Click for Gradient Space-->'

fig_velocity = go.Figure()
fig_velocity.update_layout(
    title=dict(text="Solution Velocity u", font=dict(size=20), y=0.9, x=0.5, xanchor='center', yanchor='top'),
    title_font_color='blue',
    xaxis_title="grid-points",
    yaxis_title="solution-values",
    font=dict(
        family="Courier New, monospace",
        size=10,
        color="RebeccaPurple"
    )
)

bivar_solution_u, bivar_solution_N = ml_util_object.getSurfacePredictions('viscosity', 3, 0.1, 0.1, 0.1, 5, 0.1, 500,
                                                                           800, 1500, 223.15, 50, 0)
fig_velocity_bivar = vis_utils_object.getSurfacePlot(bivar_solution_u, 'sol_u', 'viscosity')
fig_velocity_bivar.update_layout(title="3D Surface plot")

fig_grad_velocity = go.Figure()
fig_grad_velocity.update_layout(
    title=dict(text="Gradient Velocity u' ", font=dict(size=20), y=0.9, x=0.5, xanchor='center', yanchor='top'),
    title_font_color='blue',
    xaxis_title="grid-points",
    yaxis_title="gradient-values",
    font=dict(
        family="Courier New, monospace",
        size=10,
        color="RebeccaPurple"
    )
)

bivar_gradient_u, bivar_gradient_N  = ml_util_object.getSurfacePredictions('viscosity', 3, 0.1, 0.1, 0.1, 5, 0.1, 500,
                                                                           800, 1500, 223.15, 50, 1)
fig_grad_velocity_bivar = vis_utils_object.getSurfacePlot(bivar_gradient_u, 'grad_u', 'viscosity')
fig_grad_velocity_bivar.update_layout(title="3D Surface plot")

fig_tension = go.Figure()
fig_tension.update_layout(
    title=dict(text="Solution Tension N", font=dict(size=20), y=0.9, x=0.5, xanchor='center', yanchor='top'),
    title_font_color='blue',
    xaxis_title="grid-points",
    yaxis_title="solution-values",
    font=dict(
        family="Courier New, monospace",
        size=10,
        color="RebeccaPurple"
    )
)

fig_tension_bivar = vis_utils_object.getSurfacePlot(bivar_solution_N, 'sol_N', 'viscosity')
fig_tension_bivar.update_layout(title="3D Surface plot")

fig_grad_tension = go.Figure()
fig_grad_tension.update_layout(
    title=dict(text="Gradient Tension N' ", font=dict(size=20), y=0.9, x=0.5, xanchor='center', yanchor='top'),
    title_font_color='blue',
    xaxis_title="grid-points",
    yaxis_title="gradient-values",
    font=dict(
        family="Courier New, monospace",
        size=10,
        color="RebeccaPurple"
    )
)

fig_grad_tension_bivar = vis_utils_object.getSurfacePlot(bivar_gradient_N, 'grad_N', 'viscosity')
fig_grad_tension_bivar.update_layout(title="3D Surface plot")

theme =  {
    'dark': True,
    'detail': '#171717',
    'primary': '#00EA64',
    'secondary': '#6E6E6E',
}

app.layout = html.Div(
                html.Div(
                    id="colors",
                    className="right-panel-controls",
                    children=daq.DarkThemeProvider(theme=theme,
                        children=[
                        html.Div([
                            html.Div
                            ([
                                html.Div(
                                    [
                                        html.Div([
                                            html.Div([
                                                html.Div(id='analysis_text', children='Click for Bivariate Analysis-->',
                                                         style={"text-align": "center", 'fontSize': 18,
                                                                'font-family': 'Courier New, monospace',
                                                                "font-weight": "bold", 'position': 'relative',
                                                                'top': '10px', 'left': '90px'})
                                                ], className='ten columns'),
                                            html.Div([
                                                daq.PowerButton(id='analysis_button', size=30, on='True', color='#FF5E5E')
                                                ], className='two columns', style={'position': 'relative',
                                                                                   'top': '5px',
                                                                                   'left': '40px'})
                                            ], className="four columns"),
                                        html.Div([
                                            html.Div(children='Melt Spinning : Isothermal',
                                                     style={"text-align": "center", 'fontSize': 28,
                                                            "font-weight": "bold",
                                                            "background-color": "yellow", "padding": "1px",
                                                            "border": "1px solid black"
                                                            })
                                            ], className="four columns"),
                                        html.Div([
                                            html.Div([
                                                html.Div(id='space_text', children='Click for Gradient Space-->',
                                                         style={"text-align": "center", 'fontSize': 18,
                                                                'font-family': 'Courier New, monospace',
                                                                "font-weight": "bold", 'position': 'relative',
                                                                'top': '10px', 'left': '-90px'})
                                            ], className='ten columns'),
                                            html.Div([
                                                daq.PowerButton(id='space_button', size=30, on='True', color='#FF5E5E')
                                            ], className='two columns', style={'position': 'relative', 'top': '5px',
                                                                               'left': '-160px'})
                                            ], className="four columns")
                                    ], className="row"),
                                    html.Div([
                                        html.Div([
                                            dcc.Graph(
                                                id="plot_u",
                                                figure=fig_velocity,
                                                style={'width': '600px', 'height': '400px'})
                                            ],
                                            className="six columns"),
                                        html.Div([
                                            dcc.Graph(
                                                id="plot_N",
                                                figure=fig_tension,
                                                style={'width': '600px', 'height': '400px'})
                                            ],
                                            className="six columns"),
                                    ], className="row"),
                                    html.Div([
                                        html.Div(style={'display': 'flex', 'flex-direction': 'column',
                                                        'align-items': 'center', "text-align": "center",
                                                        'font-family': 'Courier New, monospace', 'fontSize': 12,
                                                        "font-weight": "bold"},
                                                 children= [
                                                    html.Div([
                                                        html.Div(id='rOut_x_text', children=[
                                                            html.Label(dl.DashLatex(r"""$rOut_x$"""))
                                                        ], className='four columns', style={'position': 'relative',
                                                                                           'top': '0px',
                                                                                           'left': '-20px',
                                                                                            "background-color": "white"}),
                                                        html.Div(id='rOut_x_slider', children=[
                                                            daq.Slider(
                                                            id='rOut_x',
                                                            min=0.0,
                                                            max=2.5,
                                                            value=0.1,
                                                            step=0.1,
                                                            handleLabel={"showCurrentValue": True,
                                                                             "label": "Value"},
                                                            size=250
                                                            )
                                                        ], className='eight columns', style={'position': 'relative',
                                                                                           'top': '10px',
                                                                                           'left': '-20px',
                                                                                            "background-color": "white"})
                                                    ], className='row'),
                                                    html.Div([
                                                        html.Div(id='rOut_y_text', children=[
                                                            html.Label(dl.DashLatex(r"""$rOut_y$"""))
                                                        ], className='four columns', style={'position': 'relative',
                                                                                           'top': '30px',
                                                                                           'left': '-20px',
                                                                                            "background-color": "white"}),
                                                        html.Div(id='rOut_y_slider', children=[
                                                            daq.Slider(
                                                            id='rOut_y',
                                                            min=0.0,
                                                            max=2.5,
                                                            value=0.1,
                                                            step=0.1,
                                                            handleLabel={"showCurrentValue": True,
                                                                             "label": "Value"},
                                                            size=250
                                                            )
                                                        ], className='eight columns', style={'position': 'relative',
                                                                                           'top': '40px',
                                                                                           'left': '-20px',
                                                                                            "background-color": "white"})
                                                    ], className='row'),
                                                    html.Div([
                                                         html.Div(id='rOut_z_text', children=[
                                                             html.Label(dl.DashLatex(r"""$rOut_z$"""))
                                                         ], className='four columns', style={'position': 'relative',
                                                                                            'top': '60px',
                                                                                            'left': '-20px',
                                                                                            "background-color": "white"}),
                                                         html.Div(id='rOut_z_slider', children=[
                                                             daq.Slider(
                                                                 id='rOut_z',
                                                                 min=0.0,
                                                                 max=2.5,
                                                                 value=0.1,
                                                                 step=0.1,
                                                                 handleLabel={"showCurrentValue": True,
                                                                              "label": "Value"},
                                                                 size=250
                                                             )
                                                         ], className='eight columns', style={'position': 'relative',
                                                                                            'top': '70px',
                                                                                            'left': '-20px',
                                                                                            "background-color": "white"})
                                                     ], className='row'),
                                                     html.Div([
                                                         html.Div(id='g_text', children=[
                                                             html.Label(dl.DashLatex(r"""$g$ $(\mathrm{ms}^{-2})$"""))
                                                         ], className='four columns', style={'position': 'relative',
                                                                                             'top': '90px',
                                                                                             'left': '-20px'}),
                                                         html.Div(id='g_slider', children=[
                                                             daq.Slider(
                                                                 id='gravity',
                                                                 min=5,
                                                                 max=15,
                                                                 value=5,
                                                                 step=0.1,
                                                                 handleLabel={"showCurrentValue": True,
                                                                              "label": "Value"},
                                                                 size=250
                                                             )
                                                         ], className='eight columns', style={'position': 'relative',
                                                                                              'top': '100px',
                                                                                              'left': '-20px'})
                                                     ], className='row'),
                                                     html.Div([
                                                         html.Div([
                                                             html.Label(dl.DashLatex(r"""$\rho_0$ $(\mathrm{kg}\,\mathrm{m}^{-3})$"""))
                                                         ], className='four columns', style={'position': 'relative',
                                                                                             'top': '120px',
                                                                                             'left': '-20px'}),
                                                         html.Div([
                                                             daq.Slider(
                                                                 id='density',
                                                                 min=800,
                                                                 max=2000,
                                                                 value=800,
                                                                 step=0.1,
                                                                 handleLabel={"showCurrentValue": True,
                                                                              "label": "Value"},
                                                                 size=250
                                                             )
                                                         ], className='eight columns', style={'position': 'relative',
                                                                                              'top': '130px',
                                                                                              'left': '-20px'})
                                                     ], className='row'),
                                                     html.Div([
                                                         html.Div(id='TIn_text', children=[
                                                             html.Label(dl.DashLatex(r"""$T_{in}$"""))
                                                         ], className='four columns', style={'position': 'relative',
                                                                                             'top': '145px',
                                                                                             'left': '-20px',
                                                                                             "background-color": "white"}),
                                                         html.Div(id='TIn_slider', children=[
                                                             daq.Slider(
                                                                 id='TIn',
                                                                 min=500,
                                                                 max=600,
                                                                 value=500,
                                                                 step=0.1,
                                                                 handleLabel={"showCurrentValue": True,
                                                                              "label": "Value"},
                                                                 size=250
                                                             )
                                                         ], className='eight columns', style={'position': 'relative',
                                                                                              'top': '160px',
                                                                                              'left': '-20px',
                                                                                              "background-color": "white"})
                                                     ], className='row')
                                                    ], className="three columns"),
                                                 html.Div(style={'display': 'flex', 'flex-direction': 'column',
                                                        'align-items': 'center', "text-align": "center",
                                                        'font-family': 'Courier New, monospace', 'fontSize': 12,
                                                        "font-weight": "bold"},
                                                 children= [
                                                     html.Div([
                                                         html.Div([
                                                             html.Label(dl.DashLatex(r"""$\mu_c$ $(\mathrm{Pa}\, \mathrm{s})$"""))
                                                         ], className='four columns', style={'position': 'relative',
                                                                                             'top': '30px',
                                                                                             'left': '-50px'}),
                                                         html.Div([
                                                             daq.Slider(
                                                                 id='mu_c',
                                                                 min=0.01,
                                                                 max=0.5,
                                                                 value=0.15,
                                                                 step=0.01,
                                                                 handleLabel={"showCurrentValue": True,
                                                                              "label": "Value"},
                                                                 size=250
                                                             )
                                                         ], className='eight columns', style={'position': 'relative',
                                                                                              'top': '30px',
                                                                                              'left': '-40px'})
                                                     ], className='row'),
                                                     html.Div([
                                                         html.Div([
                                                             html.Label(dl.DashLatex(r"""$B$"""))
                                                         ], className='four columns', style={'position': 'relative',
                                                                                             'top': '60px',
                                                                                             'left': '-70px'}),
                                                         html.Div([
                                                             daq.Slider(
                                                                 id='B',
                                                                 min=1500,
                                                                 max=2500,
                                                                 value=1500,
                                                                 step=1,
                                                                 handleLabel={"showCurrentValue": True,
                                                                              "label": "Value"},
                                                                 size=250
                                                             )
                                                         ], className='eight columns', style={'position': 'relative',
                                                                                              'top': '70px',
                                                                                              'left': '-40px'})
                                                     ], className='row'),
                                                     html.Div([
                                                         html.Div([
                                                             html.Label(dl.DashLatex(
                                                                 r"""$T_{vf}$"""))
                                                         ], className='four columns', style={'position': 'relative',
                                                                                             'top': '90px',
                                                                                             'left': '-70px'}),
                                                         html.Div([
                                                             daq.Slider(
                                                                 id='T_VF',
                                                                 min=223.15,
                                                                 max=283.15,  # max=32462
                                                                 value=223.15,
                                                                 step=1,
                                                                 handleLabel={"showCurrentValue": True,
                                                                              "label": "Value"},
                                                                 size=250
                                                             )
                                                         ], className='eight columns', style={'position': 'relative',
                                                                                              'top': '100px',
                                                                                              'left': '-40px'})
                                                     ], className='row'),
                                                     html.Div([
                                                         html.Div([
                                                             html.Label(dl.DashLatex(r"""$u_{in}$ (dimless)"""))
                                                         ], className='four columns', style={'position': 'relative',
                                                                                             'top': '120px',
                                                                                             'left': '-50px'}),
                                                         html.Div([
                                                             daq.Slider(
                                                                 id='u_in',
                                                                 min=0.1,
                                                                 max=1,
                                                                 value=0.1,
                                                                 step=0.1,
                                                                 handleLabel={"showCurrentValue": True,
                                                                              "label": "Value"},
                                                                 size=250
                                                             )
                                                         ], className='eight columns', style={'position': 'relative',
                                                                                              'top': '130px',
                                                                                              'left': '-40px'})
                                                     ], className='row'),
                                                     html.Div([
                                                         html.Div([
                                                             html.Label(dl.DashLatex(r"""$u_{out}$ (dimless)"""))
                                                         ], className='four columns', style={'position': 'relative',
                                                                                             'top': '150px',
                                                                                             'left': '-50px'}),
                                                         html.Div([
                                                             daq.Slider(
                                                                 id='u_out',
                                                                 min=3,
                                                                 max=84,
                                                                 value=10.2,
                                                                 step=0.1,
                                                                 handleLabel={"showCurrentValue": True,
                                                                              "label": "Value"},
                                                                 size=250
                                                             )
                                                         ], className='eight columns', style={'position': 'relative',
                                                                                              'top': '160px',
                                                                                              'left': '-40px'})
                                                     ], className='row')
                                                 ], className="three columns"),
                                        html.Div([
                                            html.Div([
                                                html.H1(children=dl.DashLatex(r"""$ \frac{du}{dx} = \frac{\mathrm{Re}}{3} \frac{N u}{\mu} $"""),
                                                        style={"text-align": "center", 'fontSize': 18,
                                                            "background-color": "yellow", "padding": "1px",
                                                            "border": "1px solid black"}),
                                                html.H1(children=dl.DashLatex(r"""$ \frac{dN}{dx} = \frac{du}{dx} - \frac{1}{\mathrm{Fr}^2}\frac{\tau_g}{u} $"""),
                                                        style={"text-align": "center", 'fontSize': 18,
                                                            "background-color": "yellow", "padding": "1px",
                                                            "border": "1px solid black"})
                                                ], className='row'),
                                            html.Div([
                                                html.Div([
                                                    html.Div([
                                                        html.Div([
                                                            html.H1(children=dl.DashLatex(r"""$ \mathrm{Re} = \frac{\rho_0 u_0 L}{\mu_0} $"""),
                                                                    style={"text-align": "center", 'fontSize': 15}),
                                                            ], className='six columns'),
                                                        html.Div([
                                                            daq.LEDDisplay(
                                                                id='Re_display',
                                                                label="",
                                                                value=20,
                                                                size=8
                                                            )
                                                            ], className='six columns', style={'display': 'flex', 'flex-direction': 'column',
                                                                                               'align-items': 'center'})
                                                        ], className='row', style={"background-color": "white",
                                                                                   "padding": "1px",
                                                                                   "border": "1px solid black"}),
                                                    html.Div([
                                                        html.Div([
                                                            html.H1(children=dl.DashLatex(r"""$ \mathrm{Fr} = \frac{u_0}{\sqrt{g L}} $"""),
                                                                    style={"text-align": "center", 'fontSize': 15})
                                                            ], className='six columns'),
                                                        html.Div([
                                                            daq.LEDDisplay(
                                                                id='Fr_display',
                                                                label="",
                                                                value=100,
                                                                size=8
                                                            )
                                                            ], className='six columns', style={'display': 'flex', 'flex-direction': 'column',
                                                                                               'align-items': 'center'})
                                                        ], className='row', style={"background-color": "white",
                                                                                   "padding": "1px",
                                                                                   "border": "1px solid black"}),
                                                    html.H1(children=dl.DashLatex(
                                                        r"""$r_{in} = (0.5, 3.0, 0.5) $"""),
                                                            style={"text-align": "center", 'fontSize': 15}),
                                                    html.Div(id='BiVarRadio', children=[
                                                        html.Div(children='Select the Parameter',
                                                                 style={"text-align": "center", 'fontSize': 10,
                                                                        "font-weight": "bold",
                                                                        }, className='row'),
                                                        html.Div([
                                                            dcc.RadioItems(options=[
                                                                {'label': dl.DashLatex(r"""$rOut_x$"""), 'value': 'rad_r_x'},
                                                                {'label': dl.DashLatex(r"""$rOut_y$"""), 'value': 'rad_r_y'},
                                                                {'label': dl.DashLatex(r"""$rOut_z$"""), 'value': 'rad_r_z'},
                                                                {'label': dl.DashLatex(r"""$g$"""), 'value': 'gravity'},
                                                                {'label': dl.DashLatex(r"""$\rho_0$"""), 'value': 'density'},
                                                                {'label': dl.DashLatex(r"""$T_{in}$"""), 'value': 'temperature'},
                                                                {'label': dl.DashLatex(r"""$\mu_c$"""), 'value': 'viscosity'},
                                                                {'label': dl.DashLatex(r"""$B$"""), 'value': 'B'},
                                                                {'label': dl.DashLatex(r"""$T_{vf}$"""), 'value': 'T_VF'},
                                                                {'label': dl.DashLatex(r"""$u_{in}$"""), 'value': 'velocity_in'},
                                                                {'label': dl.DashLatex(r"""$u_{out}$"""), 'value': 'velocity_out'}
                                                            ], value='viscosity', id='radio_param', inline=True)
                                                        ], style={"text-align": "center", 'fontSize': 10,
                                                                  "font-weight": "bold", 'font-family': 'Courier New, monospace'},
                                                            className='row')
                                                        ], style={'display': 'none'}, className='row')
                                                    ], className='six columns', style={"background-color": "white",
                                                                                       "padding": "1px",
                                                                                       "border": "1px solid black"}),
                                                html.Div([
                                                    html.H1(children=dl.DashLatex(
                                                        r"""$u_0 = 1 \mathrm{ms}^{-1}, \mu_0 = 1 \mathrm{Pa}\, \mathrm{s} $"""),
                                                            style={"text-align": "center", 'fontSize': 15}),
                                                    html.Div([
                                                        html.Div([
                                                            html.H1(children=dl.DashLatex(
                                                                r"""$ L = \frac{r_{out} - r_{in} }{\lVert r_{out} - r_{in} \rVert} $"""),
                                                                    style={"text-align": "center", 'fontSize': 15})
                                                            ], className='six columns'),
                                                        html.Div([
                                                            daq.LEDDisplay(
                                                                id='L_display',
                                                                label="",
                                                                value=100,
                                                                size=8
                                                            )
                                                            ], className='six columns', style={'display': 'flex', 'flex-direction': 'column',
                                                                                               'align-items': 'center'})
                                                        ], className='row', style={"background-color": "white",
                                                                                   "padding": "1px",
                                                                                   "border": "1px solid black"}),
                                                    html.Div([
                                                        html.Div([
                                                            html.H1(children=dl.DashLatex(
                                                                r"""$ \tau_g = \frac{r_{out} - r_{in} }{L} * eg $"""),
                                                                    style={"text-align": "center", 'fontSize': 15})
                                                        ], className='six columns'),
                                                        html.Div([
                                                            daq.LEDDisplay(
                                                                id='tau_g_display',
                                                                label="",
                                                                value=100,
                                                                size=8
                                                            )
                                                        ], className='six columns',
                                                            style={'display': 'flex', 'flex-direction': 'column',
                                                                   'align-items': 'center'})
                                                    ], className='row', style={"background-color": "white",
                                                                               "padding": "1px",
                                                                               "border": "1px solid black"}),
                                                    html.Div([
                                                        html.Div([
                                                            html.H1(children=dl.DashLatex(
                                                                r"""$ \mu = \mu_c * \exp(\frac{B}{T_{in} - T_{vf}}) $"""),
                                                                    style={"text-align": "center", 'fontSize': 13})
                                                        ], className='six columns'),
                                                        html.Div([
                                                            daq.LEDDisplay(
                                                                id='mu_display',
                                                                label="",
                                                                value=100,
                                                                size=8
                                                            )
                                                        ], className='six columns',
                                                            style={'display': 'flex', 'flex-direction': 'column',
                                                                   'align-items': 'center'})
                                                    ], className='row', style={"background-color": "white",
                                                                               "padding": "1px",
                                                                               "border": "1px solid black"})
                                                ], className='six columns', style={"background-color": "white",
                                                                                       "padding": "1px",
                                                                                       "border": "1px solid black"})
                                                ], className='row')
                                            ], className="six columns")
                                        ], className="row")
                                ])
                            ])
                        ])
                    )
                )

@app.callback(
    [dash.dependencies.Output('Re_display', 'value'),
     dash.dependencies.Output('Fr_display', 'value')],
    [dash.dependencies.Input('gravity', 'value'),
     dash.dependencies.Input('density', 'value'),
     dash.dependencies.Input('L_display', 'value')]
)
def update_Re_Fr(g, rho, L):
    return round(rho*L, 3), round(1/np.sqrt(g*L), 3)

@app.callback(
    [dash.dependencies.Output('L_display', 'value'),
     dash.dependencies.Output('tau_g_display', 'value'),
     dash.dependencies.Output('mu_display', 'value')],
    [dash.dependencies.Input('rOut_x', 'value'),
     dash.dependencies.Input('rOut_y', 'value'),
     dash.dependencies.Input('rOut_z', 'value'),
     dash.dependencies.Input('mu_c', 'value'),
     dash.dependencies.Input('B', 'value'),
     dash.dependencies.Input('T_VF', 'value'),
     dash.dependencies.Input('TIn', 'value')]
)
def update_L_tau_mu(rOut_x, rOut_y, rOut_z, mu_c, B, T_VF, TIn):
    rOut = np.array([rOut_x, rOut_y, rOut_z])
    len = LA.norm(rOut - np.array([0.5, 3.0, 0.5]))
    tau = (rOut - np.array([0.5, 3.0, 0.5])) / len
    tau_g = -tau[1]
    mu = mu_c * np.exp(B / (TIn - T_VF))
    return round(len, 3), round(tau_g, 3), round(mu, 3)

@app.callback(
    [dash.dependencies.Output('analysis_text', 'children'),
     dash.dependencies.Output('space_text', 'children')],
    [dash.dependencies.Input('analysis_button', 'on'),
     dash.dependencies.Input('space_button', 'on')]
)
def updateAnalysisAndSpace(analysis, space):
    global analysis_text_global, space_text_global

    if analysis:
        analysis_text_global = 'Click for Bivariate Analysis-->'
    else:
        analysis_text_global = 'Click for Univariate Analysis-->'

    if space:
        space_text_global = 'Click for Gradient Space-->'
    else:
        space_text_global = 'Click for Solution Space-->'

    return analysis_text_global, space_text_global

@app.callback(
    [dash.dependencies.Output('plot_u', 'figure'),
     dash.dependencies.Output('plot_N', 'figure'),
     dash.dependencies.Output('BiVarRadio', 'style')],
    [dash.dependencies.Input('space_button', 'on'),
     dash.dependencies.Input('analysis_button', 'on'),
     dash.dependencies.Input('rOut_x', 'value'),
     dash.dependencies.Input('rOut_y', 'value'),
     dash.dependencies.Input('rOut_z', 'value'),
     dash.dependencies.Input('gravity', 'value'),
     dash.dependencies.Input('density', 'value'),
     dash.dependencies.Input('TIn', 'value'),
     dash.dependencies.Input('mu_c', 'value'),
     dash.dependencies.Input('B', 'value'),
     dash.dependencies.Input('T_VF', 'value'),
     dash.dependencies.Input('u_in', 'value'),
     dash.dependencies.Input('u_out', 'value'),
     dash.dependencies.Input('radio_param', 'value')
     ]
)
def updateSpaceGraph(space, analysis, r_x, r_y, r_z, g, rho, t_in, mu_c, b, t_vf, u_in, u_out, radio):
    global bivar_solution_u, bivar_solution_N, bivar_gradient_u, bivar_gradient_N
    if space:
        if analysis:
            predictions = ml_util_object.getPredictions(np.array([[u_out, r_x, r_y, r_z, g, u_in, t_in, rho, mu_c, b, t_vf]]))
            res_u, res_N = ml_util_object.getResidual(np.array([[u_out, r_x, r_y, r_z, g, u_in, t_in, rho, mu_c, b, t_vf]]))
            fig_velocity = vis_utils_object.getSolutionGraph(predictions[:, 0], np.linspace(0.0, 1.0, num=100), res_u, 'u')
            fig_tension = vis_utils_object.getSolutionGraph(predictions[:, 1], np.linspace(0.0, 1.0, num=100), res_N, 'N')
            return fig_velocity, fig_tension, {'display': 'None'}
        else:
            if radio == 'velocity_out':
                bivar_solution_u, bivar_solution_N = ml_util_object.getSurfacePredictions(radio, r_x, r_y, r_z, g, u_in,
                                                                                          t_in, rho, mu_c, b, t_vf, 50, 0)
            elif radio == 'rad_r_x':
                bivar_solution_u, bivar_solution_N = ml_util_object.getSurfacePredictions(radio, u_out, r_y, r_z, g, u_in,
                                                                                          t_in, rho, mu_c, b, t_vf, 50, 0)
            elif radio == 'rad_r_y':
                bivar_solution_u, bivar_solution_N = ml_util_object.getSurfacePredictions(radio, u_out, r_x, r_z, g, u_in,
                                                                                          t_in, rho, mu_c, b, t_vf, 50, 0)
            elif radio == 'rad_r_z':
                bivar_solution_u, bivar_solution_N = ml_util_object.getSurfacePredictions(radio, u_out, r_x, r_y, g, u_in,
                                                                                          t_in, rho, mu_c, b, t_vf, 50, 0)
            elif radio == 'gravity':
                bivar_solution_u, bivar_solution_N = ml_util_object.getSurfacePredictions(radio, u_out, r_x, r_y, r_z, u_in,
                                                                                          t_in, rho, mu_c, b, t_vf, 50, 0)
            elif radio == 'velocity_in':
                bivar_solution_u, bivar_solution_N = ml_util_object.getSurfacePredictions(radio, u_out, r_x, r_y, r_z, g,
                                                                                          t_in, rho, mu_c, b, t_vf, 50, 0)
            elif radio == 'temperature':
                bivar_solution_u, bivar_solution_N = ml_util_object.getSurfacePredictions(radio, u_out, r_x, r_y, r_z, g,
                                                                                          u_in, rho, mu_c, b, t_vf, 50, 0)
            elif radio == 'density':
                bivar_solution_u, bivar_solution_N = ml_util_object.getSurfacePredictions(radio, u_out, r_x, r_y, r_z, g,
                                                                                          u_in, t_in, mu_c, b, t_vf, 50, 0)
            elif radio == 'viscosity':
                bivar_solution_u, bivar_solution_N = ml_util_object.getSurfacePredictions(radio, u_out, r_x, r_y, r_z, g,
                                                                                          u_in, t_in, rho, b, t_vf, 50, 0)
            elif radio == 'B':
                bivar_solution_u, bivar_solution_N = ml_util_object.getSurfacePredictions(radio, u_out, r_x, r_y, r_z, g,
                                                                                          u_in, t_in, rho, mu_c, t_vf, 50, 0)
            else:
                bivar_solution_u, bivar_solution_N = ml_util_object.getSurfacePredictions(radio, u_out, r_x, r_y, r_z, g,
                                                                                          u_in, t_in, rho, mu_c, b, 50, 0)
            return vis_utils_object.getSurfacePlot(bivar_solution_u, 'sol_u', radio), \
                   vis_utils_object.getSurfacePlot(bivar_solution_N, 'sol_N', radio), {'display': 'inherit'}
    else:
        if analysis:
            du, dN = ml_util_object.getGradients(np.array([[u_out, r_x, r_y, r_z, g, u_in, t_in, rho, mu_c, b, t_vf]]))
            fig_grad_velocity = vis_utils_object.getGradientGraph(du, np.linspace(0.0, 1.0, num=100), 'u')
            fig_grad_tension = vis_utils_object.getGradientGraph(dN, np.linspace(0.0, 1.0, num=100), 'N')
            return fig_grad_velocity, fig_grad_tension, {'display': 'None'}
        else:
            if radio == 'velocity_out':
                bivar_gradient_u, bivar_gradient_N  = ml_util_object.getSurfacePredictions(radio, r_x, r_y, r_z, g, u_in,
                                                                                          t_in, rho, mu_c, b, t_vf, 50, 1)
            elif radio == 'rad_r_x':
                bivar_gradient_u, bivar_gradient_N  = ml_util_object.getSurfacePredictions(radio, u_out, r_y, r_z, g,
                                                                                          u_in,
                                                                                          t_in, rho, mu_c, b, t_vf, 50, 1)
            elif radio == 'rad_r_y':
                bivar_gradient_u, bivar_gradient_N  = ml_util_object.getSurfacePredictions(radio, u_out, r_x, r_z, g,
                                                                                          u_in,
                                                                                          t_in, rho, mu_c, b, t_vf, 50, 1)
            elif radio == 'rad_r_z':
                bivar_gradient_u, bivar_gradient_N  = ml_util_object.getSurfacePredictions(radio, u_out, r_x, r_y, g,
                                                                                          u_in,
                                                                                          t_in, rho, mu_c, b, t_vf, 50, 1)
            elif radio == 'gravity':
                bivar_gradient_u, bivar_gradient_N  = ml_util_object.getSurfacePredictions(radio, u_out, r_x, r_y, r_z,
                                                                                          u_in,
                                                                                          t_in, rho, mu_c, b, t_vf, 50, 1)
            elif radio == 'velocity_in':
                bivar_gradient_u, bivar_gradient_N  = ml_util_object.getSurfacePredictions(radio, u_out, r_x, r_y, r_z,
                                                                                          g,
                                                                                          t_in, rho, mu_c, b, t_vf, 50, 1)
            elif radio == 'temperature':
                bivar_gradient_u, bivar_gradient_N  = ml_util_object.getSurfacePredictions(radio, u_out, r_x, r_y, r_z,
                                                                                          g,
                                                                                          u_in, rho, mu_c, b, t_vf, 50, 1)
            elif radio == 'density':
                bivar_gradient_u, bivar_gradient_N  = ml_util_object.getSurfacePredictions(radio, u_out, r_x, r_y, r_z,
                                                                                          g,
                                                                                          u_in, t_in, mu_c, b, t_vf, 50, 1)
            elif radio == 'viscosity':
                bivar_gradient_u, bivar_gradient_N  = ml_util_object.getSurfacePredictions(radio, u_out, r_x, r_y, r_z,
                                                                                          g,
                                                                                          u_in, t_in, rho, b, t_vf, 50, 1)
            elif radio == 'B':
                bivar_gradient_u, bivar_gradient_N  = ml_util_object.getSurfacePredictions(radio, u_out, r_x, r_y, r_z,
                                                                                          g,
                                                                                          u_in, t_in, rho, mu_c, t_vf,
                                                                                          50, 1)
            else:
                bivar_gradient_u, bivar_gradient_N  = ml_util_object.getSurfacePredictions(radio, u_out, r_x, r_y, r_z,
                                                                                          g,
                                                                                          u_in, t_in, rho, mu_c, b, 50, 1)
            return vis_utils_object.getSurfacePlot(bivar_gradient_u, 'grad_u', radio), \
                   vis_utils_object.getSurfacePlot(bivar_gradient_N, 'grad_N', radio), {'display': 'inherit'}

if __name__ == '__main__':
	app.run_server(debug=False)