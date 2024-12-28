import unittest
from bls.PTF import PTF

class TestPTF(unittest.TestCase):
    def setUp(self):
        # Set up parameters for the PTF class
        self.param = {
            "D": 4,
            "ptf": "max"
        }
        self.ptf = PTF(self.param)

    def test_initialization(self):
        # Test if the PTF object is initialized correctly
        self.assertEqual(self.ptf.param, self.param)
        self.assertEqual(self.ptf.D2, 2 * self.param["D"])
        self.assertEqual(self.ptf.K, math.ceil(math.log2(self.ptf.D2)))
        self.assertEqual(len(self.ptf.wp), self.param["D"])
        self.assertEqual(len(self.ptf.wn), self.param["D"])
        self.assertEqual(self.ptf.local_weights, [1] * self.ptf.D2)
        self.assertEqual(self.ptf.local_threshold, self.ptf.D2 / 2)
        self.assertEqual(self.ptf.y, 0)
        self.assertEqual(self.ptf.x, [0] * self.ptf.D2)

    def test_set_min(self):
        # Test the SetMin method
        self.ptf.SetMin()
        # Add assertions to verify the state after calling SetMin

    def test_set_max(self):
        # Test the SetMax method
        self.ptf.SetMax()
        # Add assertions to verify the state after calling SetMax

    def test_set_medmax(self):
        # Test the SetMedMax method
        self.ptf.SetMedMax()
        # Add assertions to verify the state after calling SetMedMax

    def test_set_median(self):
        # Test the SetMedian method
        self.ptf.SetMedian()
        # Add assertions to verify the state after calling SetMedian

if __name__ == "__main__":
    unittest.main()