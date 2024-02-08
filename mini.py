import boto3


def send_to_firehose(stream_name, data):
    firehose_client = boto3.client('firehose')
    data_byte_string = str(data).encode('utf-8')
    try:
        response = firehose_client.put_record(
            DeliveryStreamName=stream_name, Record={'Data': data_byte_string})
        return response
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


while True:
    print(send_to_firehose(
        "d-firehose-cw-metrics-to-elastic", {
            "metric_stream_name": "TEst",
            "account_id": "765985352288",
            "region": "eu-west-1",
            "namespace": "AWS/KinesisAnalytics",
            "metric_name": "cpuUtilization",
            "dimensions": {
                "Application": "d-flink-realtime-pipeline"
            },
            "timestamp": 1707211320000,
            "value": {
                "max": 6.0,
                "min": 4.0,
                "sum": 18.0,
                "count": 4.0
            },
            "unit": "None"
        }))
    