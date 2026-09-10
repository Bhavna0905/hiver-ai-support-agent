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
                "response_tweet_id": "3",
                "in_response_to_tweet_id": 1,
            },
            {
                "tweet_id": 3,
                "author_id": "customer_1",
                "inbound": True,
                "created_at": "Tue Oct 31 22:12:47 +0000 2017",
                "text": "Thanks, I will DM you.",
                "response_tweet_id": "4",
                "in_response_to_tweet_id": 2,
            },
            {
                "tweet_id": 4,
                "author_id": "amazonhelp",
                "inbound": False,
                "created_at": "Tue Oct 31 22:13:47 +0000 2017",
                "text": "Great, we will help you there.",
                "response_tweet_id": None,
                "in_response_to_tweet_id": 3,
            },
        ]
    )

    conversations = build_conversations(
        df,
        brand_name="amazonhelp",
    )

    assert len(conversations) == 1

    conversation = conversations[0]

    assert conversation["customer_id"] == "customer_1"
    assert conversation["turn_count"] == 4

    tweet_ids = [
        message["tweet_id"]
        for message in conversation["messages"]
    ]

    assert tweet_ids == [1, 2, 3, 4]


def test_multiple_customers_are_not_merged():
    """
    Two customers reply within the same interaction.
    They must remain separate conversations.
    """
    df = pd.DataFrame(
        [
            {
                "tweet_id": 1,
                "author_id": "customer_1",
                "inbound": True,
                "created_at": "Tue Oct 31 22:10:47 +0000 2017",
                "text": "My order is late",
                "response_tweet_id": "2",
                "in_response_to_tweet_id": None,
            },
            {
                "tweet_id": 2,
                "author_id": "amazonhelp",
                "inbound": False,
                "created_at": "Tue Oct 31 22:11:47 +0000 2017",
                "text": "Please send us a DM.",
                "response_tweet_id": "3",
                "in_response_to_tweet_id": 1,
            },
            {
                "tweet_id": 3,
                "author_id": "customer_1",
                "inbound": True,
                "created_at": "Tue Oct 31 22:12:47 +0000 2017",
                "text": "I already sent a DM.",
                "response_tweet_id": None,
                "in_response_to_tweet_id": 2,
            },
            {
                "tweet_id": 4,
                "author_id": "customer_2",
                "inbound": True,
                "created_at": "Tue Oct 31 22:13:47 +0000 2017",
                "text": "I also need help with my refund.",
                "response_tweet_id": "5",
                "in_response_to_tweet_id": 2,
            },
            {
                "tweet_id": 5,
                "author_id": "amazonhelp",
                "inbound": False,
                "created_at": "Tue Oct 31 22:14:47 +0000 2017",
                "text": "Please send us a DM.",
                "response_tweet_id": None,
                "in_response_to_tweet_id": 4,
            },
        ]
    )

    conversations = build_conversations(
        df,
        brand_name="amazonhelp",
    )

    customer_ids = {
        conversation["customer_id"]
        for conversation in conversations
    }

    assert "customer_1" in customer_ids
    assert "customer_2" in customer_ids


def test_customer_reply_is_associated_with_brand():
    """
    A customer tweet whose parent is a brand tweet should
    be associated with that brand.
    """
    df = pd.DataFrame(
        [
            {
                "tweet_id": 1,
                "author_id": "amazonhelp",
                "inbound": False,
                "in_response_to_tweet_id": None,
                "response_tweet_id": "2",
            },
            {
                "tweet_id": 2,
                "author_id": "customer_1",
                "inbound": True,
                "in_response_to_tweet_id": 1,
                "response_tweet_id": None,
            },
        ]
    )

    brand_tweet_to_brand = {
        1: "amazonhelp",
    }

    parent_id = df.iloc[1]["in_response_to_tweet_id"]

    assert int(parent_id) in brand_tweet_to_brand

    assert (
        brand_tweet_to_brand[int(parent_id)]
        == "amazonhelp"
    )