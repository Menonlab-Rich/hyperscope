import unittest
import numpy as np
from ml_models.base.dataset import GenericDataLoader


class MockDataLoader(GenericDataLoader):
    def get_ids(self):
        return np.arange(100)

    def __getitem__(self, idx):
        return idx


class TestGenericDataLoader(unittest.TestCase):
    def setUp(self):
        # Create a dummy dataset with 100 IDs
        self.dataset = MockDataLoader()
        # second dataset for testing reproducibility of seed
        self.dataset2 = MockDataLoader()

    def test_len(self):
        self.assertEqual(len(self.dataset), 100)

    def test_getitem(self):
        # Test getting an item by index
        item = self.dataset[0]
        self.assertIsNotNone(item)

    def test_iter(self):
        # Test iterating over the dataset
        count = 0
        for item in self.dataset:
            count += 1
        self.assertEqual(count, 100)

    def test_split(self):
        # Test splitting the dataset into training and validation sets
        train_ids, val_ids = self.dataset.split(train_ratio=0.8, seed=16)
        train_ids2, val_ids2 = self.dataset2.split(train_ratio=0.8, seed=16)
        self.assertEqual(len(train_ids), 80)
        self.assertEqual(len(val_ids), 20)
        self.assertTrue(np.all(train_ids == train_ids2))

    def test_fold(self):
        # Ensure both datasets have the same IDs before folding
        self.assertTrue(np.array_equal(
            self.dataset.get_ids(),
            self.dataset2.get_ids()))
        # Test creating k folds for cross-validation
        folds = self.dataset.fold(k=5, shuffle=True)
        folds2 = self.dataset2.fold(k=5, shuffle=True)

        # Ensure the correct number of folds are created
        self.assertEqual(len(folds), 5)
        
        # Ensure the same folds are created for both datasets
        for (train_ids, val_ids), (train_ids2, val_ids2) in zip(folds, folds2):
            self.assertIsNotNone(train_ids)
            self.assertIsNotNone(val_ids)
            self.assertIsNotNone(train_ids2)
            self.assertIsNotNone(val_ids2)
            self.assertTrue(np.array_equal(train_ids, train_ids2))
            self.assertTrue(np.array_equal(val_ids, val_ids2))


if __name__ == '__main__':
    unittest.main()
import unittest
from ml_models.resnet.dataset import ResnetDataModule


class TestResnetDataModule(unittest.TestCase):
    def setUp(self):
        # Create an instance of ResnetDataModule
        self.data_module = ResnetDataModule()

    def test_train_dataloader(self):
        # Test the train_dataloader method
        train_dataloader = self.data_module.train_dataloader()
        self.assertIsNotNone(train_dataloader)

    def test_val_dataloader(self):
        # Test the val_dataloader method
        val_dataloader = self.data_module.val_dataloader()
        self.assertIsNotNone(val_dataloader)

    def test_test_dataloader(self):
        # Test the test_dataloader method
        test_dataloader = self.data_module.test_dataloader()
        self.assertIsNone(test_dataloader)

    def test_predict_dataloader(self):
        # Test the predict_dataloader method
        predict_dataloader = self.data_module.predict_dataloader()
        self.assertIsNotNone(predict_dataloader)

    def test_setup_fit(self):
        # Test the setup method for 'fit' stage
        self.data_module.setup('fit')
        train_set = self.data_module.train_set
        val_set = self.data_module.val_set
        self.assertIsNotNone(train_set)
        self.assertIsNotNone(val_set)

    def test_setup_test(self):
        # Test the setup method for 'test' stage
        self.data_module.setup('test')
        test_set = self.data_module.test_set
        self.assertIsNone(test_set)

    def test_setup_predict(self):
        # Test the setup method for 'predict' stage
        self.data_module.setup('predict')
        prediction_set = self.data_module.prediction_set
        self.assertIsNotNone(prediction_set)


if __name__ == '__main__':
    unittest.main()