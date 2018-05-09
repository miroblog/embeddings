from matplotlib import pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

size_list = [50, 100, 300, 500, 1000]
window_list = [2, 5, 7, 10]
min_count_list = [10, 20, 50, 100]

param_list = {
    'size': size_list,
    'window': window_list,
    'count': min_count_list
}

def select_dataframe(df, model, type):
    return df[(df['model'] == model) & (df['type'] == type)]

def compare_param(df, pivot_param):
    type = df['type'].get_values()[0]
    model = df['model'].get_values()[0]

    fig, ax1 = plt.subplots()
    series_list = []
    name_list = []
    for param in param_list[pivot_param]:
        selected_df = df[df[pivot_param] == param]
        selected_df = selected_df.reset_index(drop=True)
        series_list.append(selected_df['val_acc'])
        name_list.append(str(param))
    result_df = pd.concat(series_list, axis=1, ignore_index=True)
    result_df.columns = [name_list]
    ax1.set_title("{0}_{1}_{2}".format(type, model, pivot_param))
    result_df.boxplot(ax = ax1, showfliers=False, showbox=True, showcaps=True)
    plt.show()

def compare_model(*args):
    fig, ax1 = plt.subplots()
    vs = []
    names = []
    for arg in args:
        v1 = arg['val_acc']
        v1 = v1.reset_index(drop=True)
        type1 = arg['type'].get_values()[0]
        model1 = arg['model'].get_values()[0]
        names.append((type1, model1))
        vs.append(v1)
    all = pd.concat(vs, axis=1, ignore_index=True)
    column_names = ["{0}_{1}".format(i[0], i[1]) for i in names]
    all.columns=[column_names]
    all.boxplot(ax = ax1, showfliers=False, showbox=True, showcaps=True)
    ax1.set_title("val_acc")
    plt.show()


def plot_3d_surf(df, size):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import numpy as np
    import pandas as pd
    from sys import argv
    type = df['type'].get_values()[0]
    model = df['model'].get_values()[0]
    selected_df = df[df['size'] == size]
    x = selected_df.as_matrix(["count"])
    y = selected_df.as_matrix(["window"])
    z = selected_df.as_matrix(["val_acc"])

    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_trisurf(x.reshape(-1),y.reshape(-1),z.reshape(-1), cmap=cm.jet, linewidth=0.1)

    max_index= np.argmax(z)
    min_index = np.argmin(z)
    ax.text(x[max_index][0], y[max_index][0], z[max_index][0], 'max %s' % (str(z[max_index])), size=10, zorder=1,
            color='k')
    ax.text(x[min_index][0], y[min_index][0], z[min_index][0], 'min %s' % (str(z[min_index])), size=10, zorder=1,
            color='k')

    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_xlabel('min count')
    ax.set_ylabel('window')
    ax.set_zlabel('val_acc')
    ax.set_title("{0} dim/{1}_{2}".format(str(size), type, model))

    plt.show()

def scatter(df, size, groupBy='window', line=False):
    size_filtered_df = df[df['size']==size]

    type = df['type'].get_values()[0]
    model = df['model'].get_values()[0]

    xs = []
    ys = []
    zs = []

    fig = pyplot.figure()
    ax = Axes3D(fig)

    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    if(groupBy is not None):
        assert groupBy in ['window', 'count']
        for param in param_list[groupBy]:
            selected_df = size_filtered_df[size_filtered_df[groupBy] == param]
            if(groupBy == "window"):
                xs.append(selected_df.as_matrix(["count"]))
                ys.append(selected_df.as_matrix(["window"]))
                zs.append(selected_df.as_matrix(["val_acc"]))
            else:
                xs.append(selected_df.as_matrix(["window"]))
                ys.append(selected_df.as_matrix(["count"]))
                zs.append(selected_df.as_matrix(["val_acc"]))

        for i, x in enumerate(xs):
            x = xs[i]
            y = ys[i]
            z = zs[i]

            pivot_order = x.argsort(axis=0)
            x = x[pivot_order].reshape(-1,1)
            y = y[pivot_order].reshape(-1,1)
            z = z[pivot_order].reshape(-1,1)

            pivot_order = y.argsort(axis=0)
            x = x[pivot_order].reshape(-1,1)
            y = y[pivot_order].reshape(-1,1)
            z = z[pivot_order].reshape(-1,1)

            if(line == True):
                ax.plot_wireframe(x, y, z, color=color_list[i], label=str(param_list[groupBy][i]))
            else:
                ax.scatter(x, y, z, label=str(param_list[groupBy][i]))
        ax.set_xlabel('min_count')
        ax.set_ylabel('window')
        ax.set_zlabel('val_acc')
        ax.set_title("{0} dim/{1}_{2}".format(str(size), type, model))
        plt.legend(loc='upper left', numpoints=1, ncol=4, fontsize=8, bbox_to_anchor=[0.1, 0], title=groupBy)
    else:
        x = size_filtered_df.as_matrix(["count"])
        y = size_filtered_df.as_matrix(["window"])
        z = size_filtered_df.as_matrix(["val_acc"])
        ax.scatter(x, y, z)
        ax.set_xlabel('min_count')
        ax.set_ylabel('window')
        ax.set_zlabel('val_acc')
        ax.set_title("{0}_{1}/{2} dim".format(type, model,str(size)))
    pyplot.show()