MERCHANT_PATTERNS = {

    "Amazon": [
        "amazon",
        "amazon pay",
        "amazon seller",
        "amazon marketplace",
        "amzn",
        "amzn mktp",
        "amz",
        "amazon.in",
        "amazon prime",
        "primevideo"
    ],

    "Uber": [
        "uber",
        "uber trip",
        "uber eats",
        "uber bv",
        "help.uber",
        "ubereats",
        "uber india"
    ],

    "Netflix": [
        "netflix",
        "netflix.com",
        "netflix india"
    ],

    "Swiggy": [
        "swiggy",
        "swiggy instamart",
        "swiggy limited",
        "swiggy blr"
    ],

    "Zomato": [
        "zomato",
        "zomato online",
        "zomato limited"
    ],

    "Flipkart": [
        "flipkart",
        "fkrt",
        "flipkart internet",
        "flipkart online"
    ],

    "Google": [
        "google",
        "google play",
        "googlepay",
        "gpay",
        "google *services",
        "youtube"
    ],

    "Apple": [
        "apple",
        "apple.com",
        "itunes",
        "icloud",
        "apple services"
    ],

    "Spotify": [
        "spotify",
        "spotify ab"
    ],

    "Microsoft": [
        "microsoft",
        "msft",
        "azure",
        "office365"
    ],

    "Steam": [
        "steam",
        "valve",
        "steampowered"
    ],

    "Paytm": [
        "paytm",
        "paytm mall",
        "paytm payments"
    ],

    "PhonePe": [
        "phonepe",
        "phone pe"
    ],

    "Razorpay": [
        "razorpay",
        "razor pay"
    ],

    "Cashfree": [
        "cashfree"
    ],

    "Myntra": [
        "myntra"
    ],

    "Blinkit": [
        "blinkit",
        "grofers"
    ],

    "BigBasket": [
        "bigbasket",
        "bbdaily"
    ],

    "Starbucks": [
        "starbucks"
    ],

    "McDonalds": [
        "mcd",
        "mcdonald",
        "mcdonalds"
    ],

    "KFC": [
        "kfc"
    ],

    "Dominos": [
        "dominos",
        "domino's"
    ],

    "IRCTC": [
        "irctc",
        "indian railway"
    ],

    "MakeMyTrip": [
        "makemytrip",
        "mmt"
    ],

    "Airbnb": [
        "airbnb"
    ],

    "Ola": [
        "ola",
        "olacabs",
        "ola cabs"
    ],

    "Jio": [
        "reliance jio",
        "jio recharge",
        "jio"
    ],

    "Airtel": [
        "airtel",
        "bharti airtel"
    ],

    "Vodafone Idea": [
        "vi",
        "vodafone",
        "idea cellular"
    ]
}

def normalize_merchant(
    merchant_name: str
):

    merchant_lower = merchant_name.lower()

    for canonical, patterns in MERCHANT_PATTERNS.items():

        for pattern in patterns:

            if pattern in merchant_lower:

                return canonical

    return merchant_name.title()