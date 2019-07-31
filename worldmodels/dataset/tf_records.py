import tensorflow as tf


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def encode_floats(features):
    package = {}
    for key, value in features.items():
        package[key] = _float_feature(value.flatten().tolist())

    example_proto = tf.train.Example(features=tf.train.Features(feature=package))
    return example_proto.SerializeToString()


def parse_random_rollouts(example_proto):
    """ used in training VAE """
    features = {
        'observation': tf.io.FixedLenFeature((64, 64, 3), tf.float32),
        'action': tf.io.FixedLenFeature((3,), tf.float32)
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    return parsed_features['observation'], parsed_features['action']


def parse_latent_stats(example_proto):
    """ used in training memory """
    features = {
        'action': tf.io.FixedLenFeature((1000, 3,), tf.float32),
        'mu': tf.io.FixedLenFeature((1000, 32,), tf.float32),
        'logvar': tf.io.FixedLenFeature((1000, 32,), tf.float32)
    }
    return tf.io.parse_single_example(example_proto, features)


def shuffle_samples(
        parse_func,
        records_list,
        batch_size,
        repeat=None,
        shuffle_buffer=5000,
        num_cpu=8,
):
    """ used in vae training """
    print('building dataset from tf records')
    files = tf.data.Dataset.from_tensor_slices(records_list)

    #  get samples from different files
    dataset = files.interleave(
        lambda x: tf.data.TFRecordDataset(x), num_parallel_calls=num_cpu, cycle_length=num_cpu
    )
    #  large buffer to smooth distirbution
    dataset = dataset.shuffle(shuffle_buffer)
    #  decode the record
    dataset = dataset.map(parse_func, num_parallel_calls=num_cpu)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat(repeat).prefetch(1)
    return iter(dataset)


def batch_episodes(parse_func, records, episode_length, num_cpu=4):
    """ used in sampling latent stats """
    # files = tf.data.TFRecordDataset(records)
    # dataset = files.map(parse_func, num_parallel_calls=4)
    files = tf.data.Dataset.from_tensor_slices(records)

    dataset = files.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        num_parallel_calls=num_cpu,
        cycle_length=num_cpu,
        block_length=episode_length
    )
    dataset = dataset.map(parse_func, num_parallel_calls=num_cpu)

    dataset = dataset.batch(episode_length)
    dataset = dataset.repeat(None)
    #dataset = dataset.batch(1)
    dataset = iter(dataset)
    return dataset


if __name__ == '__main__':
    #  reading a record for manual inspection

    from worldmodels.dataset.upload_to_s3 import S3
    # s3 = S3()
    # s3_records = s3.list_all_objects('latent-stats')
    s3_records = ['/Users/adam/world-models-experiments/latent-stats/episode3.tfrecord']
    raw_dataset = tf.data.TFRecordDataset(s3_records[0])
    dataset = raw_dataset.map(parse_latent_stats)

    for num, data in enumerate(dataset.take(2)):
        pass
