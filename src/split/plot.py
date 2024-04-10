import matplotlib.pyplot as plt


def plot(dataset):
    malignant = dataset[dataset['Diagnosis'] == 'M']
    benign = dataset[dataset['Diagnosis'] == 'B']

    fig, axs = plt.subplots(
        nrows=6,
        ncols=5,
        figsize=(
            15,
            10
        ),
        tight_layout=True
    )

    for i in range(6):
        for j in range(5):
            # Computing feature index
            feature_index = i * 5 + j + 2
            feature_name = dataset.columns[feature_index]

            # Displaying malignant
            malignant_value = malignant[feature_name]
            axs[i, j].hist(
                malignant_value,
                bins=20, alpha=0.5, label='Malignant', color='red'
            )

            # Displaying benign
            benign_value = benign[feature_name]
            axs[i, j].hist(
                benign_value,
                bins=20, alpha=0.5, label='Benign', color='blue'
            )

            # Displaying title and legend
            axs[i, j].set_title(feature_name, fontsize=8)
            axs[i, j].legend(loc='upper right', fontsize=8)

    plt.show()
