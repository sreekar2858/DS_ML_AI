import marimo

__generated_with = "0.14.15"
app = marimo.App(width="medium")


@app.cell
def _():
    # --- utilities 
    import os
    import shutil
    import yaml
    from box import Box
    import marimo as mo
    # --- data pre-processing
    import numpy as np
    import polars as pl
    # --- dataset retrieve and training 
    import kagglehub as kh
    import tensorflow_datasets as tfds
    import torchvision as tv
    # --- visualization
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.express as px
    from bokeh.plotting import figure, show
    import bokeh.plotting as bp
    import bokeh.io as bio
    from bokeh.models import ColumnDataSource
    from bokeh.layouts import gridplot
    return (
        Box,
        bio,
        bp,
        figure,
        go,
        gridplot,
        kh,
        make_subplots,
        mo,
        np,
        os,
        pl,
        shutil,
        tv,
        yaml,
    )


@app.cell
def _(yaml):
    with open("./data/dataset_info.yaml", "r") as file:
        DATASETS = yaml.safe_load(file)
    return (DATASETS,)


@app.cell
def _(DATASETS):
    DATASETS.keys()
    return


@app.cell
def _(Box, DATASETS):
    dataset = Box(DATASETS)
    return (dataset,)


@app.cell
def _(dataset, kh, os, pl, shutil, tv):
    level='beginner'
    if level.lower() == 'beginner':
        for dataset_name, values in dataset.beginner.items():
            source = values.source.lower()
            target_dir = os.path.join(os.path.join(dataset.datadir, level), dataset_name)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            if not len(os.listdir(target_dir)) > 0:
                # Download from the source based on the config file
                # Currently supported sources: Kaggle, URL, torchvision
                if 'kaggle' in source:
                    path = kh.dataset_download(values.handle, 
                                               path=getattr(values, "path", None) or None)
                    shutil.move(path, target_dir)
                    print(f"Downloaded {values.handle} dataset from Kaggle to {target_dir}")
                elif 'url' in source:
                    sep = "\t" if 'tab' in getattr(values, 'separator') else ','
                    _ = pl.read_csv(values.url, separator=sep, has_header=values.header)
                    _.write_csv(os.path.join(target_dir, f"{values.handle}.csv"), separator=',')
                    print(f"Downloaded {values.handle} dataset from URL to {target_dir}")
                elif 'torchvision' in source:
                    transform = tv.transforms.Compose([tv.transforms.ToTensor(),])
                    _ = getattr(tv.datasets, values.handle)(root=target_dir, train=True, download=True, transform=transform)
                    _  = getattr(tv.datasets, values.handle)(root=target_dir, train=False, download=True, transform=transform)
                    print(f"Downloaded {values.handle} dataset from torchvision to {target_dir}")
                else:
                    raise ValueError(f"Unsupported source: {source}. Supported sources are: Kaggle, URL, torchvision")
            else:
                print(f"Dataset {dataset_name} already exists in {target_dir}, skipping download.")
    else:
        raise ValueError(f"Unsupported level: {level}. Supported levels are: beginner")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""## **Dataset: Boston Housing**""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    There are 14 attributes in each case of the dataset. They are:

        **CRIM** - per capita crime rate by town  
        **ZN** - proportion of residential land zoned for lots over 25,000 sq.ft.  
        **INDUS** - proportion of non-retail business acres per town.  
        **CHAS** - Charles River dummy variable (1 if tract bounds river; 0 otherwise)  
        **NOX** - nitric oxides concentration (parts per 10 million)  
        **RM** - average number of rooms per dwelling  
        **AGE** - proportion of owner-occupied units built prior to 1940  
        **DIS** - weighted distances to five Boston employment centres  
        **RAD** - index of accessibility to radial highways  
        **TAX** - full-value property-tax rate per $10,000  
        **PTRATIO** - pupil-teacher ratio by town  
        **B** - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town  
        **LSTAT** - % lower status of the population  
        **MEDV** - Median value of owner-occupied homes in $1000's
    """
    )
    return


@app.cell
def _(dataset, pl):
    feature_names = [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
        "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
    ]

    _ = pl.read_csv(
        f"{dataset.datadir}/beginner/Boston_Housing/boston_housing.csv",
        has_header=False,
        separator="\n"
    )
    cleaned = []
    for j in range(_.shape[0]):
        cleaned.append(_[j].to_numpy()[0][0].strip().split())

    boston_df = pl.DataFrame(cleaned, schema=feature_names, orient="row")
    boston_df = boston_df.with_columns([pl.col(c).cast(pl.Float64) for c in boston_df.columns])
    return (boston_df,)


@app.cell
def _(boston_df):
    print(boston_df.describe())
    return


@app.cell
def _():
    # fig=go.Figure()
    # fig.add_trace(go.Heatmap(x=boston_df.columns,
    #                          y=boston_df.columns,
    #                          z=boston_df.corr(),
    #                           colorscale='RdBu'
    #                         ))
    # fig.update_layout(xaxis_title="Features",
    #                   yaxis_title="Features",
    #                   title="Correlation Matrix",
    #                   width=600,
    #                   height=600
    #                  )
    # fig.show()
    return


@app.cell
def _(boston_df):
    cols = boston_df.columns
    n_features = len(cols)
    nrows, ncols = 3, 5  # for Boston Housing's 13 columns you could use 3 rows x 5 cols
    return cols, ncols, nrows


@app.cell(disabled=True, hide_code=True)
def _(boston_df, cols, go, make_subplots, ncols, nrows):
    fig = make_subplots(rows=nrows,
                        cols=ncols,
                        subplot_titles=cols,
                        vertical_spacing=0.05,
                        horizontal_spacing=0.05,
                       )

    for i, col in enumerate(cols):
        row = i // ncols + 1
        colnum = i % ncols + 1
        fig.add_trace(
            go.Box(y=boston_df[col].to_numpy(), name=col, boxmean=True, showlegend=False),
            row=row, col=colnum
        )

    fig.update_layout(
        width = 1200, height = 1200,
        title_text="Boxplots for Each Feature (Subplot Grid)",
        showlegend=False
    )
    fig.show()
    return


@app.cell
def _(bio, boston_df, bp, gridplot, np):
    def _():

        cols = boston_df.columns
        n_features = len(cols)

        # Create individual boxplot figures
        plots = []

        for col in cols:
            # Convert to numpy for boxplot calculations
            data = boston_df[col].to_numpy()

            # Calculate boxplot statistics
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            median = np.percentile(data, 50)
            iqr = q3 - q1
            upper = q3 + 1.5 * iqr
            lower = q1 - 1.5 * iqr

            # Find outliers
            outliers = data[(data > upper) | (data < lower)]

            # Create figure
            p = bp.figure(
                title=col,
                width=200,
                height=200,
                x_range=[0, 1],
                tools="pan,wheel_zoom,box_zoom,reset,save"
            )

            # Draw box
            p.quad(top=q3, bottom=q1, left=0.2, right=0.8, 
                   fill_color="lightblue", line_color="black", alpha=0.5)

            # Draw median line
            p.line([0.2, 0.8], [median, median], line_color="red", line_width=2)

            # Draw whiskers
            p.line([0.5, 0.5], [q3, min(data.max(), upper)], line_color="black")
            p.line([0.5, 0.5], [q1, max(data.min(), lower)], line_color="black")
            p.line([0.4, 0.6], [min(data.max(), upper), min(data.max(), upper)], line_color="black")
            p.line([0.4, 0.6], [max(data.min(), lower), max(data.min(), lower)], line_color="black")

            # Add outliers if any
            if len(outliers) > 0:
                p.scatter([0.5] * len(outliers), outliers, size=4, color="red", alpha=0.6)

            # Style the plot
            p.xaxis.visible = False
            p.title.text_font_size = "12pt"
            p.title.align = "center"

            plots.append(p)

        # Arrange plots in a grid (3 rows x 5 cols for 14 features)
        grid = gridplot(plots, ncols=5, sizing_mode="scale_width")
        # Show the plot
        bio.show(grid)
        return

    _()
    return


@app.cell(disabled=True, hide_code=True)
def _(boston_df, cols, go, make_subplots, ncols, nrows):
    def _():
        # Create a distribution plot of the features
        fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=cols,
                            vertical_spacing=0.1, horizontal_spacing=0.05)

        for i, col in enumerate(cols):
            row = i // ncols + 1
            colnum = i % ncols + 1
            fig.add_trace(
                go.Histogram(x=boston_df[col].to_numpy(), name=col, showlegend=False),
                row=row, col=colnum
            )
            fig.update_layout(
                width=1200, height=1200,
                title_text="Distribution of Each Feature (Subplot Grid)",
                showlegend=False,
                # update the layout for better visibility
                margin=dict(l=20, r=20, t=40, b=20),
                template="plotly_white"
            )
            fig.update_xaxes(title_text="Value")
            fig.update_yaxes(title_text="Count")
        return fig.show()


    _()
    return


@app.cell
def _(bio, boston_df, cols, figure, gridplot, np):
    def _():
        # Render in notebook
        bio.output_notebook()
        plots = []
        for col in cols:
            hist, edges = np.histogram(boston_df[col], bins=30, density=True)
            p = figure(
                title=f"Average {col}",
                x_axis_label=col,
                y_axis_label="Density",
                width=200,
                height=200,
                tools="pan,wheel_zoom,box_zoom,reset,save",
                # tooltip
                tooltips=[(col, "@x"), ("Density", "@top")],
                toolbar_location="above"           
            )
            p.quad(
                top=hist, bottom=0, left=edges[:-1], right=edges[1:],
                fill_color="skyblue", line_color="white"
            )
            # Draw a line for the probability distribution function
            # Probability density function
            x = np.linspace(-4, 4, 100)
            pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
            p.line(x, pdf, line_width=2, line_color="navy")
        
            plots.append(p)
            # Arrange plots in a grid (3 rows x 5 cols for 14 features)
            grid = gridplot(plots, ncols=5, sizing_mode="scale_width")
            # Show the plot
        return bio.show(grid)


    _()
    return


if __name__ == "__main__":
    app.run()
