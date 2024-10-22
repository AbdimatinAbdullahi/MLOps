import json
import requests

# API Gateway endpoint URL
api_gateway_url = "https://11esr06t73.execute-api.us-east-1.amazonaws.com/dev/EmployeeData"

# Employee data to be sent for prediction
employee_data = {
    "feature1": 0.43,
    "feature2": 151,
    "feature3": 0,
    "feature4": 1
}

# Convert the employee data to JSON
payload = json.dumps(employee_data)

def send_data_for_prediction(data):
    try:
        # Send a POST request to the API Gateway
        response = requests.post(api_gateway_url, data=data, headers={
            "Content-Type": "application/json"
        })
        
        # Check the response status
        if response.status_code == 200:
            prediction = response.json()
            print("Prediction is: ", prediction)
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print("Error occurred: ", str(e))

# Call the function with the payload
send_data_for_prediction(payload)
