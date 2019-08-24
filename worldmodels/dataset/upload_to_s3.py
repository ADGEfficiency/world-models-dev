
# class S3:
#     """ interface for AWS S3 """
#     def __init__(self, bucket='world-models'):
#         self.bucket_name = bucket

#         self.client = boto3.client(
#             's3',
#             region_name='eu-central-1',
#             config=Config(signature_version=UNSIGNED)
#         )

#     def post(self, fname, key=None):
#         if key is None:
#             key = fname
#         print('uploading {} to s3'.format(fname))
#         self.client.upload_file(fname, self.bucket_name, key)

#     def get(self, key, fname):
#         print('getting {} from S3'.format(key))
#         return self.client.download_file(self.bucket_name, key, fname)

