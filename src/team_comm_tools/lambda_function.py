import json

def lambda_handler(event, context):
    try:
        # Open and read the filtered_dict.json file
        with open('filtered_dict.json', 'r') as json_file:
            filtered_dict = json.load(json_file)
        
        # Return the filtered_dict in the response body
        return {
            'statusCode': 200,
            'body': json.dumps(filtered_dict),
            'headers': {
                'Content-Type': 'application/json'
            }
        }
    
    except Exception as e:
        # Handle exceptions and return an error message
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': 'Internal Server Error',
                'error': str(e)
            }),
            'headers': {
                'Content-Type': 'application/json'
            }
        }