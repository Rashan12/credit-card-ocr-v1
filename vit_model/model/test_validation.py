from validation import validate_card_details

# Test validation
card_details = {
    "card_number": "4283 9820 0208 7152",
    "expiry_date": "04/26"
}

result = validate_card_details(card_details)
print(result)