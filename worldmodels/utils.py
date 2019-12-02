import os

import boto3


#  not importing from params nice not to have the dependency for now
home = os.environ['HOME']
results_dir = os.path.join(home, 'world-models-experiments')


def make_directories(*dirs):
    [os.makedirs(os.path.join(results_dir, d), exist_ok=True) for d in dirs]


def calc_batch_per_epoch(
    epochs, batch_size, records, samples_per_record=1
):
    """ used in vae & memory training """
    print('training of {} epochs'.format(epochs))
    batch_per_epoch = int(samples_per_record * len(records) / batch_size)
    print('{} batches per epoch'.format(batch_per_epoch))
    return epochs, batch_size, batch_per_epoch


def list_records(
    path, contains, data
):
    """ interface to S3 or local files """
    if str(data).lower() == 's3':
        return list_s3_objects(contains)

    elif data == 'local':
        return list_local_files(path, contains)

    else:
        raise ValueError('data source {} not recognized'.format(data))


def list_s3_objects(contains):
    print('S3 objects that include {}'.format(contains))
    s3 = boto3.resource('s3')
    name = 'world-models'
    bucket = s3.Bucket(name)
    objs = bucket.objects.all()
    objs = [o for o in objs if contains in o.key]
    print('found {} objects'.format(objs))
    return sorted(['s3://{}/{}'.format(name, o.key) for o in objs])


def list_local_files(record_dir, incl):
    print('local files that contain {} in {}'.format(incl, record_dir))
    record_dir = os.path.join(results_dir, record_dir)
    files = os.listdir(record_dir)
    files = sorted([os.path.join(record_dir, f) for f in files if incl in f])
    print('found {} files'.format(len(files)))
    return files
