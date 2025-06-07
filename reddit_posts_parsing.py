import praw
import time
import pandas as pd
from datetime import datetime
from tqdm import tqdm

def parse_subreddit_posts(subreddit_name, limit=None):
    subreddit = reddit.subreddit(subreddit_name)
    all_posts = []
    with tqdm(total=limit, desc="Parsing posts", unit="post") as pbar:
        post_counter = 0
        for post in subreddit.new(limit=None):
            try:
                post_data = {
                    'title': post.title,
                    'url': post.url,
                    'post': post.selftext.strip(),
                    'date': datetime.fromtimestamp(post.created_utc).strftime("%d %B %Y")
                }

                all_posts.append(post_data)

                post_counter += 1
                pbar.update(1)

                if post_counter % 60 == 0:
                    time.sleep(2)

                if post_counter % 100 == 0:
                    time.sleep(5)
            
            except Exception as e:
                print(f'Can not process post: {e}')
                continue

        return all_posts

reddit = praw.Reddit(
    client_id="client_id",
    client_secret="client_secret",
    username="username",
    password="password",
    user_agent="user_agent"
)

subreddit_name = 'regretfulparents'
posts = parse_subreddit_posts(subreddit_name)

df = pd.DataFrame(posts)
df.set_index('date', inplace=True)
df.to_csv('regretful_parents_posts.csv')
