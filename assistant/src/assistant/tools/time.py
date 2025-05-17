def get_time(iana_timezone: str) -> str:
    """
    Get the current time for a given IANA timezone.
    """
    from datetime import datetime
    import pytz

    # Get the current time in the specified timezone
    tz = pytz.timezone(iana_timezone)
    current_time = datetime.now(tz)

    # Format the time as a string in 12-hour format
    formatted_time = current_time.strftime("%I:%M %p")
    return formatted_time