import json
import boto3
import os

sagemaker_client = boto3.client("sagemaker-runtime")
endpoint_name = os.environ["Endpoint_name"]

def lambda_handler(event, context):
    try:
        # Read the incoming request body
        data = event
        
        # Prepare the payload for the SageMaker endpoint
        payload = json.dumps([[data['feature1'], data['feature2'], data['feature3'], data['feature4']]])
        
        # Invoke the SageMaker endpoint
        response = sagemaker_client.invoke_endpoint(
            EndpointName=endpoint_name,
            Body=payload,
            ContentType="application/json"
        )
        
        # Read and decode the response
        prediction = response['Body'].read().decode()
        
        return {
            'statusCode': 200,
            'body': json.dumps({'prediction': prediction})
        }
    
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
