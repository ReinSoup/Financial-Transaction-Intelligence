CATEGORY_MAPPING = {

    # Shopping
    "Amazon": "Shopping",
    "Flipkart": "Shopping",
    "Myntra": "Shopping",

    # Food
    "Swiggy": "Food",
    "Zomato": "Food",
    "McDonalds": "Food",
    "KFC": "Food",
    "Dominos": "Food",
    "Starbucks": "Food",

    # Transport
    "Uber": "Transport",
    "Ola": "Transport",
    "IRCTC": "Transport",
    "MakeMyTrip": "Travel",
    "Airbnb": "Travel",

    # Subscriptions
    "Netflix": "Subscriptions",
    "Spotify": "Subscriptions",
    "YouTube": "Subscriptions",
    "Apple": "Subscriptions",

    # Telecom
    "Jio": "Utilities",
    "Airtel": "Utilities",
    "Vodafone Idea": "Utilities",

    # Payments / Finance
    "Paytm": "Finance",
    "PhonePe": "Finance",
    "Razorpay": "Finance",
    "Cashfree": "Finance",

    # Tech / Software
    "Google": "Technology",
    "Microsoft": "Technology",
    "Steam": "Gaming",

    # Grocery
    "Blinkit": "Groceries",
    "BigBasket": "Groceries"
}

def map_category(
    merchant: str
):

    return CATEGORY_MAPPING.get(
        merchant,
        "Other"
    )