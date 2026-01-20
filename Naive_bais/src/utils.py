import os
import matplotlib.pyplot as plt

def plot_platform_sentiment(df, results_dir):
    filtered = df[df['predicted_label'].isin(['positive', 'negative'])]

    platform_counts = (
        filtered
        .groupby(['Platform', 'predicted_label'])
        .size()
        .unstack(fill_value=0)
    )

    platform_counts.plot(kind='bar', figsize=(8,5))
    plt.title("Platform-wise Positive vs Negative Sentiment")
    plt.xlabel("Platform")
    plt.ylabel("Tweet Count")
    plt.xticks(rotation=0)
    plt.legend(title="Sentiment")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "platform_sentiment_bar.png"))
    plt.close()
