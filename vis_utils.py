import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

class VisUtils:
    def __init__(self):
        print('VisInit')
        # implementation of init goes here


    def getSolutionGraph(self, solution, grid_points, residual, name):
        fig_solution = go.Figure()
        fig_solution.add_trace(go.Scatter(x=grid_points, y=solution, mode='lines', name='solution'))
        if name=='u':
            fig_solution.update_layout(
                title=dict(text="Solution Velocity u", font=dict(size=40), y=0.9, x=0.5, xanchor='center', yanchor='top'),
                title_font_color='blue',
                xaxis_title="grid-points",
                yaxis_title="solution-values",
                xaxis=dict(range=[0, 1.02]),
                font=dict(
                    family="Courier New, monospace",
                    size=18,
                    color="RebeccaPurple"
                )
            )
        else:
            fig_solution.update_layout(
                title=dict(text="Solution Velocity N", font=dict(size=40), y=0.9, x=0.5, xanchor='center',
                           yanchor='top'),
                title_font_color='blue',
                xaxis_title="grid-points",
                yaxis_title="solution-values",
                xaxis=dict(range=[-0.01, 1.02]),
                font=dict(
                    family="Courier New, monospace",
                    size=18,
                    color="RebeccaPurple"
                )
            )

        return fig_solution

    def getGradientGraph(self, gradient, grid_points, name):
        fig_gradient = go.Figure()
        fig_gradient.add_trace(go.Scatter(x=grid_points, y=gradient, mode='lines', name=''))
        if name=='u':
            fig_gradient.update_layout(
                title=dict(text="Gradient Velocity u", font=dict(size=40), y=0.9, x=0.5, xanchor='center', yanchor='top'),
                title_font_color='blue',
                xaxis_title="grid-points",
                yaxis_title="gradient-values",
                font=dict(
                    family="Courier New, monospace",
                    size=18,
                    color="RebeccaPurple"
                )
            )
        else:
            fig_gradient.update_layout(
                title=dict(text="Gradient Velocity N", font=dict(size=40), y=0.9, x=0.5, xanchor='center',
                           yanchor='top'),
                title_font_color='blue',
                xaxis_title="grid-points",
                yaxis_title="gradient-values",
                font=dict(
                    family="Courier New, monospace",
                    size=18,
                    color="RebeccaPurple"
                )
            )

        return fig_gradient

    def getSurfacePlot(self, dataframe, str, var):
        fig = go.Figure(data=[go.Surface(z=dataframe.values.tolist(), y=np.array(dataframe.index),
                                         x=np.array(dataframe.columns))])
        if str == 'sol_u':
            fig.update_layout(
                scene=dict(
                    xaxis_title="grid-points",
                    yaxis_title=var,
                    zaxis_title='solution-velocity (u)',
                    camera=dict(eye=dict(x=1.4, y=-1.3, z=-0.2))),
                font=dict(
                    family="Courier New, monospace",
                    size=14,
                    color="RebeccaPurple"),
                margin=dict(l=0, r=0, t=0, b=0)
            )
        elif str == 'sol_N':
            fig.update_layout(
                scene=dict(
                    xaxis_title="grid-points",
                    yaxis_title=var,
                    zaxis_title='solution-tension (N)',
                    camera=dict(eye=dict(x=1.4, y=-1.3, z=-0.2))),
                font=dict(
                    family="Courier New, monospace",
                    size=14,
                    color="RebeccaPurple"),
                margin=dict(l=0, r=0, t=0, b=0)
            )
        elif str == 'grad_u':
            fig.update_layout(
                scene=dict(
                    xaxis_title="grid-points",
                    yaxis_title=var,
                    zaxis_title='gradient-velocity (u)',
                    camera=dict(eye=dict(x=1.4, y=-1.3, z=-0.2))),
                font=dict(
                    family="Courier New, monospace",
                    size=14,
                    color="RebeccaPurple"),
                margin=dict(l=0, r=0, t=0, b=0)
            )
        else:
            fig.update_layout(
                scene=dict(
                    xaxis_title="grid-points",
                    yaxis_title=var,
                    zaxis_title='gradient-tension (N)',
                    camera=dict(eye=dict(x=1.4, y=-1.3, z=-0.2))),
                font=dict(
                    family="Courier New, monospace",
                    size=14,
                    color="RebeccaPurple"),
                margin=dict(l=0, r=0, t=0, b=0)
            )
        return fig
