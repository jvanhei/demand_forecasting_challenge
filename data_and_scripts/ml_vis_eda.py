#!/home/jvanh/anaconda3/bin/python
"""
.. module:: import
   :platform: Linux
   :synopsis: library of python functions for data visualization and exploratory data analysis (EDA)
 
.. author:: Jan van Heiningen
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
# 

def pd_read_csv_stats_describe(fname=None, in_df=None):
    """ 
    Read a csv file or an existing pandas dataframe and print statistics
        
    :param fname: [str] (optional) filename of csv -- If None then 'in_df' is required
    :param in_df: [pandas.core.frame.DataFrame] (optional) pandas dataframe

    :return: [None] prints out summary and statistics
    """
    if fname is not None:
        in_df = pd.read_csv(fname)
    elif in_df is None:
        raise ValueError('if fname is None, then a dataframe must be supplied in "in_df"')
    num = 10
    print(f'\n{fname} - data in first {num} rows')
    print(in_df.head(num))
    desc = in_df.describe(include='all').T
    desc['zero count'] = (in_df == 0).sum()
    desc['nan count'] = in_df.isna().sum()
    desc['unique count'] = in_df.nunique()
    print(f'\n{fname} - summary of column statistics')
    print(desc)
    print()
    df_shape = in_df.shape
    if len(df_shape) == 2:
        print(f'\n{fname} - correlation matrix (numeric columns only)')
        print(in_df.corr(numeric_only=True, method='pearson').round(2))
    print(in_df.info())
    if fname is not None:
        return in_df

 
def plot_scatter_xy(xvals, xlabel, yvals, ylabel, no_data, ax, targets=None, gmap=None, marker_size=2):
    """
    Create an x-y scatter-plot for and color each bar according to the targets value

    Parameters:
    - xvals (numpy array vector of x-values): num data points (rows)
    - xlabel: string xlabel 
    - yvals (numpy array vector of y-values): num data points (rows)
    - ylabel: string ylabel 
    - no_data (int or float): Value to exclude as invalid (default: -1).
    - targets (1D numpy array): Target values used to color the bars.
    - ax: handle to the figure plot
    - cmap: mpl.colors.LinearSegmentedColormap object, e.g. mpl.cm.cool or a solid color e.g. 'blue'
    - marker_size: (int): size of marker in 1/72 of an inch
    Returns:
    - None
    """
    dataxy, mask = mask_data(np.c_[xvals, yvals], no_data, True)
    max_points = 10000
    num_points = dataxy.shape[0]
    np.random.seed(0)
    interval = int(num_points / max_points) + 1
    dataxy = dataxy[::interval]
    num_points = dataxy.shape[0]
    perm = np.random.permutation(np.arange(num_points))
    dataxy = dataxy[perm]
    if type(gmap) == mpl.cm.ScalarMappable: 
        colors = gmap.to_rgba(targets)[::interval][perm]
    else:
        colors = gmap
    ax.scatter(dataxy[:, 0], dataxy[:, 1], c=colors, marker='.', s=marker_size)

    # Set x-axis and y-axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    

def plot_multiple(data2d, xlabels, ylabel=None, targets=None, nbins=100, target_label='', no_data=np.nan, ax_size=4., range_targets=None, cmap=None, marker_size=2):
    """
    Create histograms for multiple features or scatterplots of 1 feature vs all the others

    Parameters:
    - data2d (2d numpy float array): N X D -- num data points (rows) X num features (columns)
    - xlabels (list): List of feature labels with length D
    - ylabel (std: must be in xlabels (this column is used as dependent variable for scatterplots instead)
    - targets (1D numpy array): Target values used to color the bars (requires cmap as mpl.colors.LinearSegmentedColormap)
    - nbins (int): Number of bins for histograms (default: 100).
    - target_label (str): Label for the target variable (default: '').
    - no_data (int or float): Value to exclude from histograms (default: -1).
    - ax_size (float): Value in inches for the width and height of each plot (default=4.)
    - range_targets: (list of (two) min/max values) if targets are specified, this overrides min/max values to saturate cmap colors
    - cmap: mpl.colors.LinearSegmentedColormap object, e.g. mpl.cm.cool or a solid color e.g. 'blue'
    - marker_size: (int): size of marker in 1/72 of an inch

    Returns:
    - None
    """
    if len(xlabels) > 0:
        assert np.issubdtype(data2d[0,0], np.floating), print(f'input data must be 2d float array data')
        nplots = len(xlabels)
        assert nplots == data2d.shape[1], print(f'Number of columns in data2d=={data2d.shape[1]} does not equal len(labels)=={nplots}')
        ncols, nrows = make_square_approx(nplots)

        fig, inv_width_bar = prep_multi_fig_bar(ax_size, ncols, nrows)
        gmap = prep_colormap(targets, range_targets, cmap)

        if ylabel is not None:
            ylabel_ind = list(xlabels).index(ylabel)
            data_y = data2d[:, ylabel_ind]

        for indx, xlabel in enumerate(xlabels):
            xrow, ycol = indx % nrows, indx // nrows
            ax = prep_subplot2grid(inv_width_bar, nrows, ncols, xrow, ycol, fig)
            data = data2d[:, indx]
            if (ylabel is not None) and (indx != ylabel_ind):
                plot_scatter_xy(data, xlabel, data_y, ylabel, no_data, ax, targets, gmap, marker_size)
            else:
                plot_histogram(data, targets, ax, gmap, nbins, no_data, xlabel)

        add_colorbar(fig, gmap, 'average ' + target_label, nrows, ncols, inv_width_bar)
        fig.set_tight_layout(True)
    else:
        print('Nothing to plot!')

def prep_multi_fig_bar(ax_size, ncols, nrows):
    fig = plt.figure(figsize=(ax_size * (ncols + 0.5), ax_size * nrows))
    inv_width_bar = int(8 / ncols) + 1
    return fig, inv_width_bar
 
def prep_subplot2grid(inv_width_bar, nrows, ncols, xrow, ycol, fig):
    ax = plt.subplot2grid((inv_width_bar*nrows, inv_width_bar*ncols+1), 
            (inv_width_bar*xrow, inv_width_bar*ycol), colspan=inv_width_bar, rowspan=inv_width_bar, fig=fig)
    return ax

def plot_histogram(data, targets, ax, gmap, nbins, no_data, xlabel):
    """
    Plot a histogram and color each bar according to the probability of the target value.

    Parameters:
    - data (array-like): Data for histogram plotting.
    - targets (array-like): Target values used to color the bars.
    - ax (Axes): Axes object to plot on.
    - gmap (ScalarMappable): ScalarMappable object for color mapping.
    - nbins (int): Number of bins for the histogram.
    - no_data (int or float): Value to exclude from histograms.
    - xlabel (str): Label for the x-axis.

    Returns:
    - None
    """
    data, mask = mask_data(data, no_data)
    unique_data = np.unique(data)
    num_data = len(unique_data)
    # categorical data is {0, 1, 2,...,C}
    categorical = (num_data < nbins) and (data.min() >= 0) and np.allclose(unique_data, np.round(unique_data), rtol=1.e-5, atol=1.e-8)
    if categorical:
        bin_width = 0.5
        min_val = unique_data.min()
        max_val = unique_data.max()
        bins_ax = np.arange(min_val-0.5*bin_width, max_val+1.5*bin_width, bin_width)  # make the bin width 0.5
        bin_vals = np.arange(int(round(min_val)), int(round(max_val))+1)
        ax.set_xticks(bin_vals, [str(val) for val in bin_vals], rotation='vertical')
    else:
        bins_ax = nbins
    _, bins_ax, patches = ax.hist(data, bins=bins_ax)
    nbins_ax = len(bins_ax) - 1 
    for ind in range(nbins_ax):
        inds = (data >= bins_ax[ind])
        if ind < nbins_ax:
            inds &= (data < bins_ax[ind+1])
        if type(gmap) == mpl.cm.ScalarMappable:
            values = targets[mask][inds]
            if len(values) > 0:
                cvalue = np.mean(values)
                patches[ind].set_facecolor(gmap.to_rgba(cvalue))
        else:
            patches[ind].set_facecolor(gmap)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('counts')

def mask_data(data, no_data, two_d=False):
    """ mask the whole row of data if there is at least 1 entry in a row that is invalid """
    if np.isnan(no_data):
        mask = ~np.isnan(data)
    else:
        mask = data != no_data
    if two_d:
        mask = np.all(mask, axis=1)
    data = data[mask]
    return data, mask

def test_mask_data():
    data = np.array([6, 0., -1, -1, 2])
    data, mask = mask_data(data, -1)
    assert np.all(data == np.array([6, 0., 2]))
    assert np.all(mask == np.array([1, 1, 0, 0, 1]))
    data = np.array([[2, 0.], [0., 0.], [3, 2.], [0, 2], [4, 4]])
    data, mask = mask_data(data, 0., True)
    assert np.all(data == np.array([[3, 2.], [4, 4]]))
    assert np.all(mask == np.array([0,0,1,0,1], dtype='bool'))
    data = np.array([[1, 2.], [np.nan, np.nan], [3, 5.], [1., np.nan], [np.nan, 4]])
    data, mask = mask_data(data, np.nan, True)
    assert np.all(data == np.array([[1, 2.], [3, 5.]]))
    assert np.all(mask == np.array([1,0,1,0,0], dtype='bool'))
    


def make_square_approx(nplots):
    """ 
    return integer valued (nrows, ncols) such that nplots ~= nrows * ncols
    Precisely: minimize 'nplots - nrows * ncols' subject to 'ncols - nrows <= 1' and 'nrows > 0'
    """
    sp = int(np.sqrt(nplots))
    if sp*sp == nplots:
        nrows, ncols = sp, sp
    elif sp*(sp+1) >= nplots:
        nrows, ncols = sp+1, sp
    else:
        nrows, ncols = sp+1, sp+1
    return ncols, nrows

def prep_colormap(targets, range_targets=None, cmap='black'):
    """ Return a mpl.cm.ScalarMappable for generating colors based on a cmap such as mpl.cm.cool"""
    if type(cmap) == mpl.colors.LinearSegmentedColormap:
        if range_targets is not None:
            min_c, max_c = range_targets
        else:
            min_c = float(np.min(targets))
            max_c = float(np.max(targets))
        norm = mpl.colors.Normalize(vmin=min_c, vmax=max_c)
        gmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        return gmap
    else:
        return cmap

def add_colorbar(fig, gmap, target_label, nrows, ncols, inv_width_bar):
    """
    Add a colorbar to the figure.

    Parameters:
    - fig (Figure): Figure object to add the colorbar.
    - gmap (ScalarMappable): ScalarMappable object for color mapping.
    - target_label (str): Label for the colorbar.
    - nrows (int): 
    - ncols (int): 
    - inv_width_bar (int): width of the colorbar is given by 'ax_size / inv_width_bar' (default=3)
    Returns:
    - None
    """
    if type(gmap) == mpl.cm.ScalarMappable:
        ax = plt.subplot2grid((inv_width_bar*nrows, inv_width_bar*ncols+1), 
            (0, inv_width_bar*ncols), colspan=1, rowspan=inv_width_bar*nrows, fig=fig)
        fig.colorbar(gmap, cax=ax, label=target_label)

def plot_corr(df, method='pearson', only_numeric=True, figsize=(12,12), cmap=mpl.cm.cool):
    """ display a correlation heatmap for the columns in the pandas dataframe df """
    corr_matrix = df.corr(method=method, numeric_only=only_numeric)
    fig, axes = plt.subplots(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap=cmap, ax=axes)

def df_plot_ts(df, tcol, tlabel, ycols, ylabels, col_keyvals={}, cat_cols=[], 
        no_data_value=np.nan, show=True, start=0, end=5, max_plots=5):
    """
    This function will plot time-series or averaged timeseries based on specified criteria
    The timeseries are averaged if they cannot be specified uniquely

    Parameter:
    - df (pandas dataframe): contains tabular data with column names including tcol, ycol, and optional xcols, col_keyvals (keys) and cat_cols
    - tcol (str): column name specifying time values
    - tlabel (str): time-label corresponding to tcol values
    - ycols (list of str): variables (y) (each gets a corresponding aligned subplot)
    - ylabels (list of str): ylabels for corresponding ycols
    - col_keyvals (dict): only plot specified data based on these column:value pairs
    - cat_cols (list of str): these columns are used to distinguish different time-series categories on the same plots
    - no_data_value: value ignored in plots
    - show (bool): show the plot
    - start (int): first timeseries to plot
    - end (int): last timeseries to plot
    - max_plots (int): maximum number of timeseries to show per plot
    
    Returns: (int, int, int) 

    Example parameters:
        1. col_keyvals = {'center_id': 55}, cat_cols = ['meal_id']
            will only plot all timeseries with center_id=55 for different 'meal_id' values
        2. col_keyvals = {category: 'Beverages', cuisine: 'Thai'}, cat_cols=None, 
            will only plot a single timeseries with category = 'Beverages' and cuisine = 'Thai'
    
    Example usage:
    df_plot_ts(df, 'week', 'time (week)', 'num_orders', 'num_orders', ycols=['checkout_price', 'base_price', 'emailer_for_promotion', 'homepage_featured'], ylabels=['Checkout Price', 'Base Price', 'Emailer for Promotion', 'Homepage Featured'], col_keyvals={'center_id': 55}, cat_cols=['meal_id'])
    """
    
    markerstyles = ['-o', '-v', '-^', '-<', '->', '-s', '-p', '-*', '-+', '-x']
    legend_adjust = 0.8
    
    # Filter DataFrame based on col_keyvals if provided
    for col, val in col_keyvals.items():
        df = df[df[col] == val]

    # If cat_cols provided, create subplots for each category
    if len(cat_cols) > 0:
        unique_categories = df.groupby(cat_cols, sort=True).size().index.values
        if len(cat_cols) > 1:  # convert array of tuples to 2d array
            unique_categories = list(map(list, unique_categories))
        else:  # convert 1d array to 2d array
            unique_categories = unique_categories.reshape(-1, 1)
    else:
        unique_categories = np.array([[0]])  # 1 category (all the data)

    # recalculate unique_categories based on start and end to plot
    nplots = len(unique_categories)
    end = min(end, nplots)
    unique_categories = unique_categories[start:end]

    for plot_num in range(int(np.ceil(len(unique_categories) / float(max_plots)))):
        fig, subs = plt.subplots(len(ycols), figsize=(15, 15))
        for i, cat_vals in enumerate(unique_categories[plot_num*max_plots:(plot_num+1)*max_plots]):
            # create averaged dataframe for repeated points in time
            cat_df = df.copy()
            for ind in range(len(cat_cols)):
                cat_df = cat_df[cat_df[cat_cols[ind]] == cat_vals[ind]]
            cat_ts = cat_df.groupby(tcol).mean(numeric_only=True) 
            min_ts = cat_ts.index.min()
            max_ts = cat_ts.index.max()

            # create labels for the timeseries (indicate any averaging)
            if len(cat_ts) < len(cat_df):
                print(f' Averaging repeated points for {cat_cols}:{cat_vals}')
                label = 'Averaged '
            else:
                label = ''
            for ind in range(len(cat_cols)):
                label += cat_cols[ind] + ':' + str(cat_vals[ind]) + '  '
            make_legend = len(label) > 0
            # Plot the time-series for each category
            for j, ycol in enumerate(ycols):
                ax = subs[j]
                ax.plot(cat_ts.index.values, cat_ts[ycol].values, markerstyles[i], label=label, linewidth=1, markersize=6, markerfacecolor='none')
                ax.set_xlabel(tlabel)
                ax.set_ylabel(ylabels[j])
                ax.set_xlim(xmin=min_ts, xmax=max_ts)
                ax.xaxis.set_minor_locator(AutoMinorLocator())  # turn on minor tick labels and gridlines
                ax.grid(visible=True, which='both', axis='both')
                if j == 0 and make_legend: # Put a legend to the right of the current axis
                    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            if col_keyvals:
                title = ', '.join(f"{col}: {val}" for col, val in col_keyvals.items())
                subs[0].set_title(title)
            if make_legend:
                fig.subplots_adjust(right=legend_adjust)
    if show:
        plt.show()
    print(f'Showing plots {start} to {end} from a total of {nplots} categories')
    return start, end, nplots


class TimeSeriesPlotter:
    def __init__(self, df):
        self.df = df

    def filter_data(self, col_keyvals):
        self.col_keyvals = col_keyvals
        for col, val in col_keyvals.items():
            self.df = self.df[self.df[col] == val]

    def plot_time_series(self, tcol, ycols, ylabels, tlabel, no_data_value=np.nan,
                         show=True, start=0, end=5, max_plots=5, cat_cols=[]):
        markerstyles = ['-o', '-v', '-^', '-<', '->', '-s', '-p', '-*', '-+', '-x']
        legend_adjust = 0.8

        # If cat_cols provided, create subplots for each category
        if len(cat_cols) > 0:
            unique_categories = self.df.groupby(cat_cols, sort=True).size().index.values
            if len(cat_cols) > 1:  # convert array of tuples to 2d array
                unique_categories = list(map(list, unique_categories))
            else:  # convert 1d array to 2d array
                unique_categories = unique_categories.reshape(-1, 1)
        else:
            unique_categories = np.array([[0]])  # 1 category (all the data)

        # recalculate unique_categories based on start and end to plot
        nplots = len(unique_categories)
        end = min(end, nplots)
        unique_categories = unique_categories[start:end]

        for plot_num in range(int(np.ceil(len(unique_categories) / float(max_plots)))):
            fig, subs = plt.subplots(len(ycols), figsize=(15, 15))
            for i, cat_vals in enumerate(unique_categories[plot_num*max_plots:(plot_num+1)*max_plots]):
                # create averaged dataframe for repeated points in time
                cat_df = self.df.copy()
                for ind in range(len(cat_cols)):
                    cat_df = cat_df[cat_df[cat_cols[ind]] == cat_vals[ind]]
                cat_ts = cat_df.groupby(tcol).mean(numeric_only=True) 
                min_ts = cat_ts.index.min()
                max_ts = cat_ts.index.max()

                # create labels for the timeseries (indicate any averaging)
                if len(cat_ts) < len(cat_df):
                    print(f' Averaging repeated points for {cat_cols}:{cat_vals}')
                    label = 'Averaged '
                else:
                    label = ''
                for ind in range(len(cat_cols)):
                    label += cat_cols[ind] + ':' + str(cat_vals[ind]) + '  '
                make_legend = len(label) > 0
                # Plot the time-series for each category
                for j, ycol in enumerate(ycols):
                    ax = subs[j]
                    ax.plot(cat_ts.index.values, cat_ts[ycol].values, markerstyles[i], label=label, linewidth=1, markersize=6, markerfacecolor='none')
                    ax.set_xlabel(tlabel)
                    ax.set_ylabel(ylabels[j])
                    ax.set_xlim(xmin=min_ts, xmax=max_ts)
                    ax.xaxis.set_minor_locator(AutoMinorLocator())  # turn on minor tick labels and gridlines
                    ax.grid(visible=True, which='both', axis='both')
                    if j == 0 and make_legend: # Put a legend to the right of the current axis
                        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                if self.col_keyvals:
                    title = ', '.join(f"{col}: {val}" for col, val in self.col_keyvals.items())
                    subs[0].set_title(title)
                if make_legend:
                    fig.subplots_adjust(right=legend_adjust)
        if show:
            plt.show()
        print(f'Showing plots {start} to {end} from a total of {nplots} categories')
        return start, end, nplots

    def plot_single_time_series(self, tcol, ycols, ylabels, tlabel,
            target_label=None, predicted_values=[], predicted_labels=[], predicted_times=[], 
            quantile_predictions=[], quantile_alphas=None, quantile_algorithm=None):
        markerstyles = ['-o', '-v', '-^', '-<', '->', '-s', '-p', '-*', '-+', '-x']
        legend_adjust = 0.8

        if target_label is not None:
            if target_label not in ycols:
                raise ValueError(f'{target_label} is not in {ycols}')
        
        num_tvals = len(np.unique(self.df[tcol].values))
        if self.df.shape[0] > num_tvals:
            raise ValueError(f'The dataframe df does not filter to unique time-series values using {self.col_keyvals}')
        
        fig, subs = plt.subplots(len(ycols), figsize=(15, 15))
        min_ts = self.df[tcol].min()
        max_ts = self.df[tcol].max()

        # Plot the time-series for each category
        for j, ycol in enumerate(ycols):
            ax = subs[j]
            label = ['data']
            ax.plot(self.df[tcol].values, self.df[ycol].values, markerstyles[0], label=label, linewidth=1, 
                    markersize=6, markerfacecolor='none')
            if target_label is not None and target_label == ycol:
                for i, (predicted_value, predicted_label, predicted_time) in enumerate(zip(
                        predicted_values, predicted_labels, predicted_times)):
                    ax.plot(predicted_time, predicted_value, markerstyles[i+1], label=predicted_label,
                            linewidth=1, markersize=6, markerfacecolor='none')
                if quantile_alphas is not None:
                    assert len(quantile_alphas) >= 2, 'quantile_alphas must have at least 2 increasing values between 0 and 1'
                    ax.fill_between(
                        predicted_time,
                        quantile_predictions[0],
                        quantile_predictions[2],
                        alpha=0.3,
                        label=f'{quantile_algorithm} {100*(quantile_alphas[2] - quantile_alphas[0]):.1f}% interval',
                    )
                    # plot the rest of the quantiles 
                    for k, (quantile_prediction, quantile_alpha) in enumerate(zip(quantile_predictions[1:-1], quantile_alphas[1:-1])):
                        ax.plot(predicted_time, quantile_prediction, markerstyles[i+2+k], 
                        label=f'{quantile_algorithm} ,{100*quantile_alpha:.1f}% quantile', 
                        linewidth=1, markersize=6, markerfacecolor='none')

            ax.set_xlabel(tlabel)
            ax.set_ylabel(ylabels[j])
            ax.set_xlim(xmin=min_ts, xmax=max_ts)
            ax.xaxis.set_minor_locator(AutoMinorLocator())  # turn on minor tick labels and gridlines
            ax.grid(visible=True, which='both', axis='both')
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if self.col_keyvals:
            title = ', '.join(f"{col}: {val}" for col, val in self.col_keyvals.items())
            subs[0].set_title(title)
        fig.subplots_adjust(right=legend_adjust)

# Example usage:
# plotter = TimeSeriesPlotter(df)
# plotter.filter_data({'center_id': 55})
# plotter.plot_time_series('week', ['checkout_price', 'base_price', 'emailer_for_promotion', 'homepage_featured'],
#                          ['Checkout Price', 'Base Price', 'Emailer for Promotion', 'Homepage Featured'], 'time (week)')
# 
