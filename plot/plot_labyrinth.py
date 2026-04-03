import numpy as np
from matplotlib import pyplot as plt

# adapted from https://github.com/markusmeister/Rosenberg-2021-Repository

def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], loc=None, title=None,
         xlim=None, ylim=None, xscale='linear', yscale='linear',
         xticks=None, yticks=None, xhide=False, yhide=False, yrot=False, yzero=False, yflip=False, 
         fmts=['g-','m--','b-.','r:'], linewidth=2, markersize=5, fillstyle='full',
         markeredgewidth=1,
         grid=False, equal=False, figsize=(5,3), axes=None):
    """
    Plot data points.
    X: an array or list of arrays
    Y: an array or list of arrays
    If Y exists then those values are plotted vs the X values
    If Y doesn't exist the X values are plotted
    xlabel, ylabel: axis labels
    legend: list of labels for each Y series
    loc: location of the legend, like 'upper right'
    title: duh
    xlim, ylim: [low,high] list of limits for the 2 axes 
    xscale, yscale: 'linear' or 'log'
    xticks, yticks: list of locations for tick marks, or None for auto ticks
    yhide: hide the y axis?
    yrot: rotate the yaxis label to horizontal?
    yzero: zero line for the y-axis?
    fmts: a list of format strings to be applied to successive Y-series
    linewidth, markersize, fillstyle, markeredgewidth: see docs
    grid: draw a grid?
    equal: use equal aspect ratio, i.e. same scale per unit on x and y axis?
    figsize: (h,v) in inches
    axes: pre-existing axes where to draw the plot
    Returns: axes for the plot
    """
    
    if not axes: # start a new figure
        fig = plt.figure(figsize=figsize, dpi=400)
        axes = plt.gca()
    
    def has_one_axis(X): # Return True if X (ndarray or list) has 1 axis
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    # axes.cla() # clears these axes
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt, linewidth=linewidth, markersize=markersize,
            	fillstyle=fillstyle,markeredgewidth=markeredgewidth)
        else:
            axes.plot(y, fmt, linewidth=linewidth, markersize=markersize,
            	fillstyle=fillstyle,markeredgewidth=markeredgewidth)
    set_axes(axes, xlabel, ylabel, legend, loc, xlim, ylim, xscale, yscale, 
             xticks, yticks, xhide, yhide, yrot, yzero, yflip, grid, equal)
    if title:
        plt.title(title)
    plt.tight_layout()

    return axes # useful if we started a new figure

def set_axes(axes, xlabel, ylabel, legend, loc, xlim, ylim, xscale, yscale, 
    	xticks, yticks, xhide, yhide, yrot, yzero, yflip, grid, equal):
    """Set the axes for matplotlib."""
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    if xlim:
        axes.set_xlim(xlim)
    else:
        axes.set_xlim(auto=True)
    if ylim:
        axes.set_ylim(ylim)
    else:
        axes.set_ylim(auto=True)
    if grid:
        axes.grid()
    if equal:
        axes.set_aspect(aspect='equal')
    if ylabel:
        if yrot:
            axes.set_ylabel(ylabel, fontsize=12, rotation=0, labelpad=15)
        else:
            axes.set_ylabel(ylabel, fontsize=12)
    if xlabel:
        axes.set_xlabel(xlabel, fontsize=12)
    axes.get_yaxis().set_visible(not yhide)
    axes.get_xaxis().set_visible(not xhide)
    if yzero:
        axes.axhline(color='black', linewidth=0.5)
    if yflip:
        axes.invert_yaxis()
    axes.tick_params(axis = 'both', which = 'major', labelsize = 10)
    axes.tick_params(axis = 'both', which = 'minor', labelsize = 9)
    if xticks:
        axes.set_xticks(xticks,minor=False); # no minor ticks
    if yticks:
        axes.set_yticks(yticks,minor=False); # no minor ticks
    if legend:
        axes.legend(legend, loc=loc)
    plt.draw()

from matplotlib.collections import LineCollection
from matplotlib import cm
import matplotlib.patches as patches

def plot_trajs(ma_wa, zs, xy_list, axs=None):
    def record_segments_dict(jax_path_vmap, xy_list):
        n_trial, trial_length = jax_path_vmap.shape
        segments = {0: [], 1: [], 2: []}
    
        for trial_idx in range(n_trial):
            trial_path = jax_path_vmap[trial_idx]
            trial_xys = xy_list[trial_idx].T
            
            start_idx = 0

            for i in range(1, trial_length):
                # If the value changes, record the current segment for the previous value
                if trial_path[i] != trial_path[start_idx] or i == trial_length - 1:
                    value = int(trial_path[start_idx])
                    segments[value].append(trial_xys[start_idx:i])  # Record the segment timestamps
                
                    start_idx = i  # Reset the start index for the new segment
        return segments
    xy_segments = record_segments_dict(zs, xy_list)

    if axs is None:
        fig, axs = plt.subplots(1, 3, figsize=(18,6), dpi=400)

    def plot_single_map(ma_wa, ax, curr_xy_segments, note="", min_length=2):
        segs_list = []
        t_list = []
        # Draw the maze outline
        plot(ma_wa[:,0], ma_wa[:,1], fmts=['k-'], equal=True, linewidth=2, yflip=True,
            xhide=True, yhide=True, axes=ax)
        re = [[-0.5,0.5,1,1],[-0.5,4.5,1,1],[-0.5,8.5,1,1],[-0.5,12.5,1,1],
            [2.5,13.5,1,1],[6.5,13.5,1,1],[10.5,13.5,1,1],
            [13.5,12.5,1,1],[13.5,8.5,1,1],[13.5,4.5,1,1],[13.5,0.5,1,1],
            [10.5,-0.5,1,1],[6.5,-0.5,1,1],[2.5,-0.5,1,1],
            [6.5,1.5,1,1],[6.5,11.5,1,1],[10.5,5.5,1,1],[10.5,7.5,1,1],
            [5.5,4.5,1,1],[5.5,8.5,1,1],[7.5,4.5,1,1],[7.5,8.5,1,1],[2.5,5.5,1,1],[2.5,7.5,1,1],
            [-0.5,2.5,3,1],[-0.5,10.5,3,1],[11.5,10.5,3,1],[11.5,2.5,3,1],[5.5,0.5,3,1],[5.5,12.5,3,1],
            [7.5,6.5,7,1]]
        for r in re:
            rect = patches.Rectangle((r[0], r[1]), r[2], r[3], linewidth=1, edgecolor='lightgray', facecolor='lightgray')
            ax.add_patch(rect)

          # Turn off the axes

        # Loop over all trajectories and collect segments and time arrays
        for xy in curr_xy_segments:
            if xy.shape[0] < min_length:
                continue
            x = -0.5 + 15 * xy[:, 0]
            y = -0.5 + 15 * xy[:, 1]
            t = np.linspace(0, 1, x.shape[0])  # Time variable from 0 to 1

            # Set up a list of (x, y) points
            points = np.array([x, y]).transpose().reshape(-1, 1, 2)

            # Set up a list of segments
            segs = np.concatenate([points[:-1], points[1:]], axis=1)

            # Collect segments and corresponding time arrays
            segs_list.append(segs)
            t_list.append(t[:-1])  # t[:-1] since segments are between points

        # Concatenate all segments and time arrays
        if not segs_list:  # No segments to plot
            return None
        all_segs = np.concatenate(segs_list)
        all_t = np.concatenate(t_list)

        # Create a single LineCollection with all segments
        lc = LineCollection(all_segs, cmap=plt.get_cmap('viridis'), linewidths=2)
        # Normalize colors to span [0, 1] fully so colorbar covers Start->End without blank
        from matplotlib.colors import Normalize
        lc.set_norm(Normalize(vmin=0.0, vmax=1.0))
        # Use segment-wise time values aligned to segments (length = n_segments)
        # all_t currently comes from t[:-1]; ensure it spans [0,1]
        if len(all_t) > 0:
            # Rescale to [0,1] explicitly in case rounding stops at <1
            tmin = float(np.min(all_t))
            tmax = float(np.max(all_t))
            if tmax > tmin:
                all_t = (all_t - tmin) / (tmax - tmin)
            else:
                all_t = np.zeros_like(all_t)
        lc.set_array(all_t)

        # Add the LineCollection to the axes
        lines = ax.add_collection(lc)
        ax.set_title(note, fontsize=24)
        return lines
    
    notes = ['water', 'home', 'explore']
    lines_list = []
    for i in range(3):
        lines = plot_single_map(ma_wa, axs[i], xy_segments[i], note=notes[i], min_length=2)
        lines_list.append(lines)
    # plt.axis('off')
    if axs is None:
        plt.show()
    else:
        return axs, lines_list


color_options = [
    (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0),
    (0.5490196078431373, 0.33725490196078434, 0.29411764705882354, 1.0),
    (0.09019607843137255, 0.7450980392156863, 0.8117647058823529, 1.0)
]

def PlotMazeWall(m_wa,axes=None,figsize=4):
    '''
    Plots the walls of the maze defined in m.
    axes: provide this to add to an existing plot
    figsize: in inches (only if axes=None)
    '''
    if axes:
        plot(m_wa[:,0],m_wa[:,1],fmts=['k-'],equal=True,linewidth=2,yflip=True,
             xhide=True,yhide=True,axes=axes) # this way we can add to an existing graph
    else:
        axes = plot(m_wa[:,0],m_wa[:,1],fmts=['k-'],equal=True,linewidth=2,yflip=True,
                  figsize=(figsize,figsize),xhide=True,yhide=True)
    return axes

import matplotlib.colors as mcolors
def PlotMazeFunction(f, state_name, m_wa, m_ru, m_xc, m_yc, numcol='cyan', figsize=4, selected_color=None, axes=None, landmarks=None):
    '''
    Plot the maze defined in m with a function f overlaid in color
    f[]: array of something as a function of place in the maze, e.g. cell occupancy
        If f is None then the shading is omitted
    m: maze structure
    numcol: color for the numbers. If numcol is None the numbers are omitted
    figsize: in inches
    selected_color: a tuple specifying the RGBA color to be used for the colormap
    Returns: the axes of the plot.
    '''
    f = normalize(f)
    if selected_color is None:
        selected_color = color_options[0]  # Default to the first color if none is selected

    col = np.array([[0, 1, 1, 1], [1, selected_color[0], selected_color[1], selected_color[2]]])
    norm = plt.Normalize(np.min(f), np.max(f))
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', [(1, 1, 1, 1), selected_color])
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])

    if axes:
        ax = axes
        PlotMazeWall(m_wa, axes=ax, figsize=figsize)
    else:
        ax = PlotMazeWall(m_wa, axes=None, figsize=figsize)

    for j, r in enumerate(m_ru):
        x = m_xc[r[-1]]; y = m_yc[r[-1]]
        if f is not None:
            ax.add_patch(patches.Rectangle((x-0.5, y-0.5), 1.0, 1.0, lw=0,
                                            fc=custom_cmap(norm(f[j])), ec='gray'))
        if numcol:
            # if state_name[:5] == 'Water' and j == 116:
            #     ax.text(x-.4, y+.1, 'Water', fontsize=7, color=numcol)
            # elif state_name[:4] == 'Home' and j == 0:
            #     ax.text(x-.4, y+.15, 'Home', fontsize=7, color=numcol)
            # else:
            #     ax.text(x-.4, y+.15, '{:d}'.format(j),fontsize=10.5, color=numcol)  # number the ends of a run
            ax.text(x-.4, y+.15, '{:d}'.format(j),fontsize=10.5, color=numcol)  # number the ends of a run
                
        # plt.colorbar(sm, ax=ax, ticks=[0, 1], fraction=0.046, pad=0.04)
        ax.set_title(state_name, fontsize=20)

        # plt.axis('off')

    if landmarks:
        import matplotlib.patheffects as pe
        for state_idx, marker_label in landmarks.items():
            lx = m_xc[m_ru[state_idx][-1]]
            ly = m_yc[m_ru[state_idx][-1]]
            ax.plot(lx, ly, marker=marker_label, markersize=14, color='white',
                    markeredgecolor='black', markeredgewidth=1.5, zorder=20)

def plot_flow_arrows(ax, q, trans_probs, m_ru, m_xc, m_yc,
                     arrow_color='black', arrow_alpha=0.6, arrow_scale=0.35,
                     highlight_states=None, highlight_color='gold'):
    """Draw greedy-policy flow arrows on the maze.

    For each state, computes the greedy action from Q-values, resolves the
    next state via the transition matrix, and draws an arrow from state
    to next-state coordinates.

    highlight_states: set/list of state indices whose arrows use highlight_color
    """
    num_states = q.shape[0]
    hl = set(highlight_states) if highlight_states else set()
    for s in range(num_states):
        a_star = np.argmax(q[s])
        ns = np.argmax(trans_probs[s, a_star, :])
        if ns == s:
            continue
        sx, sy = float(m_xc[m_ru[s][-1]]), float(m_yc[m_ru[s][-1]])
        nx, ny = float(m_xc[m_ru[ns][-1]]), float(m_yc[m_ru[ns][-1]])
        dx, dy = nx - sx, ny - sy
        length = np.sqrt(dx**2 + dy**2)
        if length < 1e-6:
            continue
        ux, uy = dx / length, dy / length
        if s in hl:
            col, alp, lw, zord, ms = highlight_color, 0.95, 3.5, 18, 16
            sc = arrow_scale * 1.4
        else:
            col, alp, lw, zord, ms = arrow_color, arrow_alpha, 1.8, 15, 10
            sc = arrow_scale
        ddx, ddy = ux * sc, uy * sc
        ax.annotate('', xy=(sx + ddx * 0.5, sy + ddy * 0.5), xytext=(sx - ddx * 0.5, sy - ddy * 0.5),
                    arrowprops=dict(arrowstyle='->,head_length=0.25,head_width=0.15',
                                    color=col, lw=lw, mutation_scale=ms),
                    alpha=alp, zorder=zord)

def plot_path_line(ax, path_states, q, trans_probs, m_ru, m_xc, m_yc,
                   color='#4dbd4d', linewidth=2.0, linestyle='--'):
    """Draw dotted lines connecting highlighted states along greedy-policy edges.

    For each state in path_states, if its greedy next-state is also in
    path_states (or is a known goal), draw a dashed line between them.
    """
    ps = set(path_states)
    drawn = set()
    for s in path_states:
        a_star = np.argmax(q[s])
        ns = int(np.argmax(trans_probs[s, a_star, :]))
        edge = (min(s, ns), max(s, ns))
        if edge in drawn:
            continue
        drawn.add(edge)
        sx = float(m_xc[m_ru[s][-1]])
        sy = float(m_yc[m_ru[s][-1]])
        nx = float(m_xc[m_ru[ns][-1]])
        ny = float(m_yc[m_ru[ns][-1]])
        ax.plot([sx, nx], [sy, ny], color=color, linewidth=linewidth,
                linestyle=linestyle, alpha=0.9, zorder=17)

def normalize(vals):
    """
    normalize to (0, max_val)
    input:
      vals: 1d array
    """
    min_val = np.min(vals)
    max_val = np.max(vals)
    return (vals - min_val) / (max_val - min_val)