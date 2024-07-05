import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns


file_path = r'C:\Users\dhrub\Downloads\prodigy ds dataset\TASK4\twitter_training.csv'

try:
    df = pd.read_csv(file_path, header=None)  # Assuming no headers in the CSV file
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit(1)
except pd.errors.EmptyDataError:
    print(f"Error: File at {file_path} is empty")
    exit(1)
except pd.errors.ParserError:
    print(f"Error: Unable to parse file at {file_path}")
    exit(1)
except Exception as e:
    print(f"Error: An unexpected error occurred: {str(e)}")
    exit(1)

# Display the first few rows to understand its structure
print(df.head())

# Step 2: Perform Sentiment Analysis
def analyze_sentiment(text):
    analysis = TextBlob(str(text))
    sentiment_score = analysis.sentiment.polarity
    if sentiment_score > 0:
        return 'positive'
    elif sentiment_score < 0:
        return 'negative'
    else:
        return 'neutral'

# Apply sentiment analysis to each comment in the 'comments' column
df['sentiment_analysis'] = df[3].apply(analyze_sentiment)  # Assuming comments are in the fourth column (index 3)

# Display the updated dataframe with sentiment analysis results
print(df[[3, 'sentiment_analysis']])

# Step 3: Visualize Sentiment Distribution
sentiment_counts = df['sentiment_analysis'].value_counts()

plt.figure(figsize=(8, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
plt.title('Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
