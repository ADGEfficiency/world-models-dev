import os

import boto3
from botocore import UNSIGNED
from botocore.client import Config


#  not importing from params
home = os.environ['HOME']
results_dir = os.path.join(home, 'world-models-experiments')


def list_local_records(record_dir, incl):
    print('finding objects on local that contain {} in {}'.format(incl, record_dir))
    record_dir = os.path.join(results_dir, record_dir)
    files = os.listdir(record_dir)
    return sorted([os.path.join(record_dir, f) for f in files if incl in f])


class S3:
    """ interface for AWS S3 """
    def __init__(self, bucket='world-models'):
        self.client = boto3.client(
            's3',
            region_name='eu-central-1',
            config=Config(signature_version=UNSIGNED)
        )
        self.bucket_name = bucket

    def post(self, fname, key=None):
        if key is None:
            key = fname
        print('uploading {} to s3'.format(fname))
        self.client.upload_file(fname, self.bucket_name, key)

    def get(self, key, fname):
        print('getting {} from S3'.format(key))
        return self.client.download_file(self.bucket_name, key, fname)

    def list_all_objects(self, contains):
        print('finding objects on S3 that contain {}'.format(contains))

        s3 = boto3.resource('s3')
        bucket = s3.Bucket('world-models')
        objs = bucket.objects.all()
        objs = [o for o in objs if contains in o.key]
        return sorted(
            ['s3://{}/{}'.format(self.bucket_name, o.key) for o in objs]
        )
