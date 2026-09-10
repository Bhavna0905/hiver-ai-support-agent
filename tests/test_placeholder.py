import pandas as pd

from src.data.conversations import build_conversations


def test_build_conversations_does_not_crash():
    df = pd.DataFrame(
        [
            {
                "tweet_id": 1,
                "author_id": "customer_1",
                "inbound": True,
                "created_at": "Tue Oct 31 22:10:47 +0000 2017",
                "text": "@amazonhelp my order is late",
                "response_tweet_id": "2",
                "in_response_to_tweet_id": None,
            },
            {
                "tweet_id": 2,
                "author_id": "amazonhelp",
                "inbound": False,
                "created_at": "Tue Oct 31 22:11:47 +0000 2017",
                "text": "Sorry about that. Please DM us.",
                "response_tweet_id": None,
                "in_response_to_tweet_id": 1,
            },
        ]
    )

    conversations = build_conversations(
        df,
        brand_name="amazonhelp",
    )

    assert isinstance(conversations, list)
    assert len(conversations) >= 1