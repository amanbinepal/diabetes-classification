import click
import pandas as pd
import altair as alt
import os


@click.command()
@click.option(
    "--train-file", default="src/objects/train_df.csv", help="Training dataset CSV file"
)
def data_visualization(train_file, out_dir="results/figures"): 
    """This function generates 4 plots that are used for EDA data visualization. 
        1. number of record bar chart by classes ("plot_class.png")
        2. numeric features' boxplot by classes ("plot_numeric_box.png")
        3. binary features' bar chart by classes ("plot_binary_bar.png")
        4. ordinal features' bar chart by classes ("plot_ordinal_bar.png")

    Args:
        train_file (string): the path to the train data set Defaults to "objects/train_df.csv".
        out_dir (str, optional): the folder to save the plots. Defaults to "../results/figures".
    """

    # ============================ read in training data
    data = pd.read_csv(train_file)

    # ============================ enable big dataset
    alt.data_transformers.enable('vegafusion')

    # ============================ group the features
    numeric_features = ["BMI"]
    binary_features = ["HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", 
                       "HeartDiseaseorAttack", "PhysActivity", "Fruits", 
                       "Veggies", "HvyAlcoholConsump", "AnyHealthcare", 
                       "NoDocbcCost", "DiffWalk", "Sex"]
    ordinal_features = ["GenHlth", "MentHlth", "PhysHlth", 
                        "Age", "Education", "Income"]

    # ============================ generate & save: class imbalance bar chart
    chart_class_bar = alt.Chart(data, title="Number of Records of Two Classes").mark_bar().encode(
        x="Diabetes_binary:N",
        y="count()"
    ).properties(
        width=150,
        height=250)

    chart_class_bar.save(os.path.join(out_dir, 'plot_class.png'), scale_factor=2.0)

    # ============================ generate & save: numeric features box plot
    chart_numeric_box = alt.Chart(data).mark_boxplot().encode(
        x=alt.X('Diabetes_binary:N', title='Diabetes (0/1)'),
        y=alt.Y(alt.repeat('row'), type='quantitative')
    ).properties(
        width=200,
        height=150
    ).repeat(
        row=numeric_features, 
    )

    chart_numeric_box.save(os.path.join(out_dir, 'plot_numeric_box.png'), scale_factor=2.0)

    # ============================ generate & save: binary features bar chart
    chart_binary_bar = alt.Chart(data).mark_bar().transform_fold(
        binary_features,
        as_=['feature', 'value']
    ).encode(
        x=alt.X('value:N', title='0 or 1'),
        y=alt.Y('mean(Diabetes_binary):Q', title='Proportion with Diabetes'),
    ).properties(
        width=150, 
        height=150
    ).facet(
        facet='feature:N',
        columns=5
    )

    chart_binary_bar.save(os.path.join(out_dir, 'plot_binary_bar.png'), scale_factor=2.0)

    # ============================ generate & save: ordinal features bar chart
    chart_ordinal_bar = alt.Chart(data).mark_bar(size=20).encode(
        x=alt.X(alt.repeat("row"), type="quantitative", sort="ascending"),
        y="count()",
        color="Diabetes_binary:N",
        column=alt.Column("Diabetes_binary:N")
    ).properties(
        width=200,
        height=150
    ).repeat(
        row=ordinal_features
    )

    chart_ordinal_bar.save(os.path.join(out_dir, 'plot_ordinal_bar.png'), scale_factor=2.0)

    click.echo('EDA figures created')


if __name__ == "__main__":
    data_visualization()
