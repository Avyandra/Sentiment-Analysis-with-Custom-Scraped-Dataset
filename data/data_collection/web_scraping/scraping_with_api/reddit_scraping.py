import praw
import pandas as pd

reddit = praw.Reddit(
    client_id="yUfzXD87v8S2p-Z7-b29iQ",
    client_secret="x9mPNxe3ALBpxbZgRRC4L4ncNsfSHw",
    password="Shahi160930",
    user_agent="RedditScraper (by u/AvyReddit)",
    username="AvyReddit",
)

#returns a pandas dataframe
def scrape_reddit(url):
    reddit_threads = []
    submission = reddit.submission(url=url)
    from praw.models import MoreComments
    #to scrape the threads as well as top level comment
    for top_level_comment in submission.comments:
        if isinstance(top_level_comment, MoreComments):
            continue
        reddit_threads.append(top_level_comment.body)

    return pd.DataFrame(reddit_threads, columns=['Reddit Threads'])

links = ['https://www.reddit.com/r/TeslaFSD/comments/1jczru5/1000mile_review_of_fsd/',
        'https://www.reddit.com/r/TeslaModel3/comments/1ib0w4i/fsd_is_really_good/',
        'https://www.reddit.com/r/TeslaModelY/comments/1d7nki6/full_self_driving_review_after_3months_of/',
        'https://www.reddit.com/r/TeslaLounge/comments/17b46l5/tried_fsd_for_first_time_today_im_shocked_at_how/',
        'https://www.reddit.com/r/TeslaLounge/comments/1hmev28/real_talk_is_fsd_13_that_good/',
         'https://www.reddit.com/r/teslamotors/comments/1eabrwb/fsd_125_earns_good_first_impressions/',
         'https://www.reddit.com/r/TeslaLounge/comments/1dghxj4/is_fsd_actually_decent/',
         'https://www.reddit.com/r/TeslaLounge/comments/1gsxe02/fsd_is_so_much_better_than_you_think/',
         'https://www.reddit.com/r/TeslaFSD/comments/1jr2c3b/tesla_has_officially_launched_its_fsd_supervised/',
         'https://www.reddit.com/r/TeslaLounge/comments/1jgaso6/someone_retested_mark_robers_selfdriving_car_test/']

# Initialize empty master DataFrame
#master_df = pd.DataFrame(columns=["Reddit Threads"])

# Go through each link and append returned DataFrame
#for url in links:
#    try:
#        temp_df = scrape_reddit(url)
#        master_df = pd.concat([master_df, temp_df], ignore_index=True)
#    except Exception as e:
#        print(f"Error scraping {url}: {e}")

#Save to CSV if needed
#master_df.to_csv("reddit_threads.csv", index=False)

links2 = ['https://www.reddit.com/r/TeslaModelY/comments/18chxm2/a_positive_tesla_review/',
          'https://www.reddit.com/r/TeslaModelY/comments/18yrmcz/my_2022_tesla_model_y_review_at_100000_miles/',
          'https://www.reddit.com/r/teslamotors/comments/u7z5q5/positive_service_center_experience/'
]
# Initialize empty master DataFrame
master_df2 = pd.DataFrame(columns=["Reddit Threads"])

# Go through each link and append returned DataFrame
for url in links2:
    try:
        temp_df = scrape_reddit(url)
        master_df2 = pd.concat([master_df2, temp_df], ignore_index=True)
    except Exception as e:
        print(f"Error scraping {url}: {e}")

#Save to CSV if needed
master_df2.to_csv("reddit_threads2.csv", index=False)