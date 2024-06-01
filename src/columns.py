def add_column_names(dataset):
    """
    Adds meaningful column names to a dataset based on pre-defined
    categories.

    Args:
        dataset (DataFrame): The DataFrame to which column names will be
            added.

    Returns:
        DataFrame: The same DataFrame but with new column names for each of
            the features.
    """
    column_names = [
        "radius",
        "texture",
        "perimeter",
        "area",
        "smoothness",
        "compactness",
        "concavity",
        "concave points",
        "symmetry",
        "fractal dimension"
    ]

    mean_columns = [
        f"mean {column_name}"
        for column_name in column_names
    ]

    standard_error_columns = [
        f"standard error {column_name}"
        for column_name in column_names
    ]

    largest_columns = [
        f"largest {column_name}"
        for column_name in column_names
    ]

    dataset.columns = (
        [
            "ID number",
            "Diagnosis"
        ] +
        mean_columns +
        standard_error_columns +
        largest_columns
    )

    return dataset
