import re
from datetime import datetime

def luhn_check(card_number):
    digits = [int(d) for d in card_number if d.isdigit()]
    if len(digits) != 16:
        return False
    checksum = 0
    is_even = False
    for digit in digits[::-1]:
        if is_even:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
        is_even = not is_even
    return checksum % 10 == 0

def detect_card_type(card_number):
    if not card_number:
        return "Unknown"
    if card_number.startswith("4"):
        return "Visa"
    elif card_number.startswith("5"):
        return "Mastercard"
    return "Unknown"

def validate_card_details(card_details, is_front=True, confidence_scores=None, confidence_threshold=0.9):
    result = {"card_number": "", "expiry_date": "", "security_number": "", "card_type": "Unknown", "errors": [], "use_vit": {}}
    print(f"Validating card details: {card_details}, Confidence scores: {confidence_scores}")

    if is_front:
        card_number = card_details.get("card_number", "").replace(" ", "")
        print(f"Processed card number: {card_number}")
        if not re.match(r'^\d{16}$', card_number):
            result["errors"].append("Card number must be exactly 16 digits")
        else:
            if not luhn_check(card_number):
                result["errors"].append("Card number failed Luhn validation")
            else:
                result["card_number"] = ' '.join([card_number[i:i+4] for i in range(0, 16, 4)])
                result["card_type"] = detect_card_type(card_number)
                if confidence_scores and confidence_scores.get("card_number", 0) >= confidence_threshold:
                    result["use_vit"]["card_number"] = True
                else:
                    result["use_vit"]["card_number"] = False

        expiry = card_details.get("expiry_date", "")
        print(f"Processed expiry date: {expiry}")
        if not re.match(r'^\d{2}/\d{2}$', expiry):
            result["errors"].append("Expiry date must be in MM/YY format")
        else:
            try:
                month, year = map(int, expiry.split('/'))
                print(f"Parsed month: {month}, year: {year}")
                if not (1 <= month <= 12):
                    result["errors"].append("Month must be between 01 and 12")
                else:
                    expiry_date = datetime.strptime(f"{month}/{year}", "%m/%y")
                    current_date = datetime.now()
                    print(f"Expiry date: {expiry_date}, Current date: {current_date}")
                    if expiry_date < current_date:
                        result["errors"].append("Card has expired")
                    else:
                        result["expiry_date"] = expiry
                        if confidence_scores and confidence_scores.get("expiry_date", 0) >= confidence_threshold:
                            result["use_vit"]["expiry_date"] = True
                        else:
                            result["use_vit"]["expiry_date"] = False
            except ValueError as e:
                result["errors"].append(f"Invalid expiry date format: {str(e)}")
    else:
        security = card_details.get("security_number", card_details.get("cvv", "")).replace(" ", "")
        print(f"Processed security number: {security}")
        if not security:
            result["errors"].append("Security number is required for back side")
        else:
            security_len = len(security)
            if security_len not in [3, 7] or not security.isdigit():
                result["errors"].append("Security number must be 3 digits (debit) or 7 digits (credit)")
            else:
                result["security_number"] = security
                if confidence_scores and confidence_scores.get("security_number", 0) >= confidence_threshold:
                    result["use_vit"]["security_number"] = True
                else:
                    result["use_vit"]["security_number"] = False

    print(f"Validation result: {result}")
    return result