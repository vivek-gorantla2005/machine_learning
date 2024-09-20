import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
movies_df = pd.read_csv("Top_100_Movies.csv")
# print(movies_df.head())

# print(movies_df.shape)
# print(movies_df.info())
# l = list(movies_df.columns)
# print(l)
movies_df.drop('imdb_link',inplace=True,axis=1)
movies_df.drop('image',inplace=True,axis=1)
movies_df.drop('imdbid',inplace=True,axis=1)
movies_df.drop('description',inplace=True,axis=1)
movies_df.drop('id',inplace=True,axis=1)

movies_df.rename(columns={'Unnamed: 0' : 'sno'},inplace=True)
print(list(movies_df.columns))

# print(movies_df.isnull().sum())

print(movies_df[movies_df['rating']>=9][['title','rating']])

movies_df['should_watch'] = movies_df['rating'].apply(lambda x:1 if x>=9 else 0)
# print(movies_df['should_watch'])

print(movies_df.head())
#data visualization
genre_avg_rating = movies_df.groupby('genre')['rating'].mean().reset_index()

# Create a bar plot
# plt.figure(figsize=(10, 6))
# sns.barplot(x='year', y='rating', data=movies_df)
# plt.xlabel('year')
# plt.ylabel('Rating')
# plt.title('year vs rating')
# plt.xticks(rotation=45)  # Rotate genre labels for better readability
# plt.show()


# plt.figure(figsize=(10, 6))
# plt.scatter(movies_df['year'], movies_df['rating'])
# plt.xlabel('Year')
# plt.ylabel('Rating')
# plt.title('Scatter Plot of Ratings by Year')
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.scatter(movies_df['year'], movies_df['rank'])
# plt.xlabel('Year')
# plt.ylabel('rank')
# plt.title('Scatter Plot of year vs rank')
# plt.show()

movies_expanded_df = movies_df.assign(genre=movies_df['genre'].str.split(',')).explode('genre')

# Create the scatter plot
# plt.figure(figsize=(12, 6))
# sns.stripplot(x='genre', y='rating', data=movies_expanded_df, jitter=True)
# plt.xlabel('Genre')
# plt.ylabel('Rating')
# plt.title('Scatter Plot of Genre vs. Rating')
# plt.xticks(rotation=45)  # Rotate genre labels for better readability
# plt.show()

numerical_columns = ['rank', 'rating', 'year', 'should_watch']

# Calculate the correlation matrix
corr_matrix = movies_df[numerical_columns].corr()

# Create the heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
# plt.title('Correlation Heatmap')
# plt.show()

print(list(movies_df['genre']))