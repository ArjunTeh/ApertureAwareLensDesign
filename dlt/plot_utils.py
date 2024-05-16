import numpy as np
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale
import plotly.colors as pcolors
from . import constants
from . import zemax_loader


def curvature_aa_surface_to_points(k, origin, max_height=10.0):
    max_height = np.minimum(max_height, 1 / np.maximum(np.abs(k), constants.FLOAT_EPSILON))
    y = np.linspace(-max_height, max_height, 201)
    if np.isclose(k, 0):
        x = np.zeros_like(y)
    else:
        x = (1 - np.sqrt(1 - np.minimum(k**2 * y**2, 1.0))) / k

    return x + origin, y


def plot_curvature_sphere(k, origin, max_height=10.0, line_args=None):
    x, y = curvature_aa_surface_to_points(k, origin, max_height)
    return go.Scatter(x=x, y=y, line=line_args)


def visualize_state(lens, lens_desc, line_args=None, fillcolor='lightblue', offset=0.0):

    if line_args == None:
        line_args = dict(color='black', width=2.0)

    color_list = pcolors.sample_colorscale('blues', (lens[:, 2] - 1) / 2.0)

    traces = []
    for i in range(lens.shape[0]):
        if lens_desc[i] == zemax_loader.ZemaxSurfaceType.STOP:
            trace = plot_stop(origin=offset, height=lens[i, 3], thickness=0.1, max_rad=1.5*lens[i, 3], line_args=line_args)
            traces.append(trace) 
        elif lens[i, 2] > (constants.DEFAULT_EXT_IOR + 1e-4):
            # draw the lens
            trace = plot_asphere_lens(lens[i],
                                      lens[i+1], offset,
                                      max_height=lens[i, 3], max_height2=lens[i+1, 3], 
                                      line_args=line_args, fillcolor=color_list[i])
            traces.append(trace) 
        offset += lens[i, 1]

    return traces


def plot_sphere_lens(k1, o1, k2, o2, max_height=10.0, max_height2=None, line_args=None, fillcolor=None):
    if max_height2 is None:
        max_height2 = max_height
    x1, y1 = curvature_aa_surface_to_points(k1, o1, max_height)
    x2, y2 = curvature_aa_surface_to_points(k2, o2, max_height2)

    x = np.concatenate([x1, x2[::-1]])
    y = np.concatenate([y1, y2[::-1]])
    x = np.concatenate([x, [x[0]]])
    y = np.concatenate([y, [y[0]]])

    return go.Scatter(x=x, y=y, line=line_args, fill='toself', fillcolor=fillcolor)


def plot_asphere_lens(lens1, lens2, offset, max_height=10.0, max_height2=None, line_args=None, fillcolor=None):
    if max_height2 is None:
        max_height2 = max_height

    ho1 = lens1[4:] if lens1.shape[0] > 4 else np.zeros(8)
    ho2 = lens2[4:] if lens2.shape[0] > 4 else np.zeros(8)

    x1, y1 = asphere_aa_surface_to_points(lens1[0], offset, ho1, max_height)
    x2, y2 = asphere_aa_surface_to_points(lens2[0], offset + lens1[1], ho2, max_height2)

    x = np.concatenate([x1, x2[::-1]])
    y = np.concatenate([y1, y2[::-1]])
    x = np.concatenate([x, [x[0]]])
    y = np.concatenate([y, [y[0]]])

    return go.Scatter(x=x, y=y, line=line_args, fill='toself', fillcolor=fillcolor)


def asphere_aa_surface_to_points(k, origin, higher_order, max_height=10.0):
    y = np.linspace(-max_height, max_height, 100)

    conic = higher_order[0]
    x = k * y**2 / (1 + np.sqrt(1 - (1 + conic) * k**2 * y**2))

    for i, ho in enumerate(higher_order[1:]):
        x += ho * y**(2*i+2)

    return x + origin, y


def plot_stop(origin, height, thickness=0.1, max_rad=None, line_args=None):
    o = origin
    h = height
    th = thickness
    mr = max_rad if max_rad is not None else 1.5 * th

    x = [o, o, None, o, o, None, o-th, o+th, None, o-th, o+th]
    y = [h, mr, None, -h, -mr, None, h, h, None, -h, -h]

    return go.Scatter(x=x, y=y, mode='lines', line=line_args)

    
def plot_zemax_state(lens_state, line_args=None, sensor_at_origin=False):
    if line_args is not None:
        legendgroup = line_args["color"] + "lenses"
    else:
        legendgroup = "lenses"
    traces = []
    sensor_pos = lens_state[:, 1].sum()
    dist = 0
    for i in range(lens_state.shape[0]):
        k = lens_state[i, 0]
        o = dist 
        ho_terms = lens_state[i, 4:]
        height = float(lens_state[i, 3]) + 0.01
        if lens_state.shape[1] > 4:
            x, y = asphere_aa_surface_to_points(k, o, ho_terms, height)
        else:
            x, y = curvature_aa_surface_to_points(k, o, height)
        x = x-sensor_pos if sensor_at_origin else x
        traces.append(go.Scatter(x=x, y=y, line=line_args, legendgroup=legendgroup))
        dist += lens_state[i, 1]
    return traces


def plot_rays_plotly(xp, vp, L=None, line_args=None, title='rays'):
    if xp.size == 0:
        return []
    if line_args is None:
        line_args = dict(color='blue')
    if L is None:
        L = np.ones((xp.shape[1],))
    rad_max = L.max()
    traces = []
    for i in range(xp.shape[1]):
        trace = go.Scatter(
            x=xp[:, i, 0],
            y=xp[:, i, 1],
            mode="lines+markers",
            line=line_args,
            opacity=0.8 * float(L[i] / rad_max),
            legendgroup=title + line_args["color"]
        )
        traces.append(trace)
    return traces


def plotly_multifigure(fig : go.Figure, traces_list, labels=None):
    if labels is None:
        labels = [str(i) for i in range(len(traces_list))]

    fig.add_traces(traces_list[0])
    fig.frames = [go.Frame(data=traces, name=str(i)) for i, traces in enumerate(traces_list)]
    fig.update_layout(dict(
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          args=[None, {"mode" : "immediate",
                                       "frame" : {"redraw": True},
                                       "transitions" : dict(duration=0, easing="linear")}],
                          method="animate",
                          )]
        )]
    ))

    slider_steps = [
        {"args" : [[i], {
            "mode" : "immediate",
            "frame" : {"redraw": True},
            "transition" : {"duration" : 0, "easing" : "linear"}}],
         "method": "animate",
         "label": labels[i]}
    for i, tra in enumerate(traces_list)]

    fig.update_layout(
        sliders=[dict(steps=slider_steps)]
    )

    return fig