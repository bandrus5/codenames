import unittest
from util.image_functions import _calculate_final_size


class TestImageFunctions(unittest.TestCase):
    def test_calculate_final_size_width(self):
        result = _calculate_final_size(1000, 2000, desired_width=500, desired_height=None)
        self.assertEqual(result, (500, 1000))
        result = _calculate_final_size(2000, 1000, desired_width=500, desired_height=None)
        self.assertEqual(result, (500, 250))

    def test_calculate_final_size_height(self):
        result = _calculate_final_size(1000, 2000, desired_width=None, desired_height=500)
        self.assertEqual(result, (250, 500))
        result = _calculate_final_size(2000, 1000, desired_width=None, desired_height=500)
        self.assertEqual(result, (1000, 500))

    def test_calculate_final_size_both(self):
        result = _calculate_final_size(1000, 2000, desired_width=500, desired_height=500)
        self.assertEqual(result, (250, 500))
        result = _calculate_final_size(1000, 2000, desired_width=300, desired_height=500)
        self.assertEqual(result, (250, 500))
        result = _calculate_final_size(1000, 2000, desired_width=200, desired_height=500)
        self.assertEqual(result, (200, 400))

        result = _calculate_final_size(1000, 2000, desired_width=500, desired_height=300)
        self.assertEqual(result, (150, 300))
        result = _calculate_final_size(1000, 2000, desired_width=500, desired_height=200)
        self.assertEqual(result, (100, 200))

        result = _calculate_final_size(2000, 1000, desired_width=500, desired_height=500)
        self.assertEqual(result, (500, 250))
        result = _calculate_final_size(2000, 1000, desired_width=300, desired_height=500)
        self.assertEqual(result, (300, 150))
        result = _calculate_final_size(2000, 1000, desired_width=200, desired_height=500)
        self.assertEqual(result, (200, 100))

        result = _calculate_final_size(2000, 1000, desired_width=500, desired_height=300)
        self.assertEqual(result, (500, 250))
        result = _calculate_final_size(2000, 1000, desired_width=500, desired_height=200)
        self.assertEqual(result, (400, 200))

