import tensorflow as tf
def validate_dataset(filenames, reader_opts=None):
    """
    Attempt to iterate over every record in the supplied iterable of TFRecord filenames
    :param filenames: iterable of filenames to read
    :param reader_opts: (optional) tf.python_io.TFRecordOptions to use when constructing the record iterator
    """
    i = 0
    for fname in filenames:
        print('validating ', fname)

        record_iterator = tf.io.tf_record_iterator(path=fname, options=reader_opts)
        try:
            for _ in record_iterator:
                i += 1
        except Exception as e:
            print('error in {} at record {}'.format(fname, i))
            print(e)


if __name__ == '__main__':
    from worldmodels.dataset.upload_to_s3 import list_local_records
    from worldmodels.dataset.tf_records import parse_random_rollouts

    records = list_local_records('random-rollouts', 'episode')

    for record in records:
        print(record)
        for _ in tf.data.TFRecordDataset(record).map(parse_random_rollouts).take(1):
            pass
