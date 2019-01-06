import unittest

from processing.input_processing import batcher

__author__ = 'fuadissa'

class BatcherTest(unittest.TestCase):
    test_input = [[1,2,3,4,5,6,7,8,9,10], [1,2,3,4,5,6,7,8,9,10]]
    expected_output_finite = [[1,2,3,4,5,6,7,8,9,10], [1,2,3,4,5,6,7,8,9,10]]
    expected_output_infinite = [[1,2,3,4,5,6,7,8,9,10, 1,2,3,4,5,6,7,8,9,10, 1,2,3,4,5,6,7,8,9,10, 1,2,3,4,5,6,7,8,9,10],
                                [1,2,3,4,5,6,7,8,9,10, 1,2,3,4,5,6,7,8,9,10, 1,2,3,4,5,6,7,8,9,10, 1,2,3,4,5,6,7,8,9,10]]
    batch_size = 4
    def test_batching_finite(self):

        full_list_1 = list()
        full_list_2 = list()
        for (list_1, list_2), new_start in batcher(self.test_input, self.batch_size, infinite=False):
            full_list_1.extend(list_1)
            full_list_2.extend(list_2)

        self.assertListEqual(self.expected_output_finite, [full_list_1, full_list_2])

    def test_batching_infinite(self):
        
        full_list_1 = list()
        full_list_2 = list()
        for index, ((list_1, list_2), new_start) in enumerate(batcher(self.test_input, self.batch_size, infinite=True)):
            full_list_1.extend(list_1)
            full_list_2.extend(list_2)
            if index >= 4:
                break
        self.assertListEqual(self.expected_output_infinite, [full_list_1, full_list_2])     