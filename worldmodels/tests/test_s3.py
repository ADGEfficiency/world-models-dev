
def test_database():
    os.makedirs('./test', exist_ok=True)

    data = np.random.rand(512, 256)
    np.save('./test/test.npy', data)

    db = S3()

    db.post('test/test.npy', 'test-np-array')

    check = db.get('test-np-array', './test/test-check.npy')

    check = np.load('test/test-check.npy')

    np.testing.assert_array_equal(data, check)

    import shutil
    shutil.rmtree('test')
