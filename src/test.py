from kpca import KernelPCA


# test code
#print(KernelPCA('../../data/1.1.txt').description())
data_folder = '../data/Mohler/'
KernelPCA(data_folder + 'raw/1.1', 25, 'RBF', 0.7, 3, True, 4, 'similarity_rouge', 'sentence').training_results()

#print(alphas)
#print(lambdas)
#print(KernelPCA('../../data/test.txt', 2, 'rbf', 0.7, 2, True, 4, 'sorensen_word', 'word').training_results())
# import unittest
#
# class TestStringMethods(unittest.TestCase):
#
#     def test_test(self):
#     def test_test(self):
#         self.assertEqual(KernelPCA(data_folder + 'raw/1.1', 25, 'poly', 3, 3, True, 4, 'similarity_rouge', 'sentence').test_test(55), 55)
#
#     def test_isupper(self):
#         self.assertTrue(KernelPCA(data_folder + 'raw/1.1', 25, 'poly', 3, 3, True, 4, 'similarity_rouge', 'sentence').test_test(55) == 55)
#         self.assertFalse(KernelPCA(data_folder + 'raw/1.1', 25, 'poly', 3, 3, True, 4, 'similarity_rouge', 'sentence').test_test(55) != 55)
#
# if __name__ == '__main__':
#     unittest.main()
