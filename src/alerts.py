def get_aqi_category(aqi):
    """Determine the AQI category based on the AQI value."""
    if aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Satisfactory'
    elif aqi <= 200:
        return 'Moderate'
    elif aqi <= 300:
        return 'Poor'
    elif aqi <= 400:
        return 'Very Poor'
    else:
        return 'Severe'

def generate_alert(aqi=None, category=None):
    """
    Generate an alert based on AQI value or category string.
    
    :param aqi: float or int, The AQI value.
    :param category: str, The AQI category (optional, takes precedence if aqi is not provided).
    :return: dict, Contains 'Alert Level', 'Health Recommendation', and 'Warning Message'.
    """
    if category is None:
        if aqi is None:
            raise ValueError("Either 'aqi' or 'category' must be provided.")
        category = get_aqi_category(aqi)
    
    # Normalize category string for matching
    cat = category.strip().lower()
    
    if cat == 'good':
        return {
            'Alert Level': 'None',
            'Health Recommendation': 'Ideal air quality for outdoor activities.',
            'Warning Message': 'Air quality is considered satisfactory, and air pollution poses little or no risk.'
        }
    elif cat == 'satisfactory':
        return {
            'Alert Level': 'Low',
            'Health Recommendation': 'Minor breathing discomfort to sensitive people.',
            'Warning Message': 'Sensitive individuals should consider reducing prolonged or heavy outdoor exertion.'
        }
    elif cat == 'moderate':
        return {
            'Alert Level': 'Moderate',
            'Health Recommendation': 'Breathing discomfort to people with lung, asthma, and heart diseases.',
            'Warning Message': 'Active children, adults, and people with respiratory disease should limit prolonged outdoor exertion.'
        }
    elif cat == 'poor':
        return {
            'Alert Level': 'High',
            'Health Recommendation': 'Breathing discomfort to most people on prolonged exposure.',
            'Warning Message': 'Everyone should reduce prolonged or heavy exertion. Avoid outdoor activities.'
        }
    elif cat == 'very poor':
        return {
            'Alert Level': 'Severe',
            'Health Recommendation': 'Respiratory illness on prolonged exposure.',
            'Warning Message': 'Avoid all physical activity outdoors. Keep windows closed and stay indoors.'
        }
    elif cat == 'severe':
        return {
            'Alert Level': 'Critical',
            'Health Recommendation': 'Affects healthy people and seriously impacts those with existing diseases.',
            'Warning Message': 'Health warning of emergency conditions. The entire population is more likely to be affected. Stay indoors.'
        }
    else:
        return {
            'Alert Level': 'Unknown',
            'Health Recommendation': 'Unknown AQI Category.',
            'Warning Message': 'Please provide a valid AQI or Category.'
        }

if __name__ == "__main__":
    # Test cases
    print("Test AQI = 45:")
    print(generate_alert(aqi=45))
    print("\nTest Category = 'Severe':")
    print(generate_alert(category='Severe'))
