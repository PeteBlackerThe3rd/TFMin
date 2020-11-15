"""
words
"""
from tf_min.graph import TensorShape


def main():
  """
  These tests verify that the tensor_shape object correctly stores, converts,
  and manipulates tensor shapes and data layouts.
  :return: None
  """

  dim_orders = [[0, 1, 2, 3],
                [0, 1, 3, 2],
                [0, 2, 1, 3],
                [0, 2, 3, 1],
                [0, 3, 1, 2],
                [0, 3, 2, 1],
                [1, 0, 2, 3],
                [1, 0, 3, 2],
                [1, 2, 0, 3],
                [1, 2, 3, 0],
                [1, 3, 0, 2],
                [1, 3, 2, 0],
                [2, 0, 1, 3],
                [2, 0, 3, 1],
                [2, 1, 0, 3],
                [2, 1, 3, 0],
                [2, 3, 0, 1],
                [2, 3, 1, 0],
                [3, 0, 1, 2],
                [3, 0, 2, 1],
                [3, 1, 0, 2],
                [3, 1, 2, 0],
                [3, 2, 0, 1],
                [3, 2, 1, 0]]

  # Test one ensure that semantic to sig and sig to semantic re-ordering
  # are the inverse of each other. This test creates all possible
  # permutations of four dimensions and verifies the functions cancel out
  test_shape = TensorShape([2, 4, 6, 8])
  print("test shape is %s" % test_shape)

  for order in dim_orders:
    test_shape.dim_order = order
    # print("Testing dim order %s\n" % order)

    test_indices = [2, 4, 8, 16]
    sig_indices = test_shape.convert_semantic_to_significance(test_indices)
    sem_indices = test_shape.convert_significance_to_semantic(sig_indices)
    if test_indices != sem_indices:
      print("sem to sig and back again test "
            "failed.")
      print("dim order = %s" % order)
      print("test order = %s" % test_indices)
      print("sig_indices = %s" % sig_indices)
      print("sem_indices = %s" % sem_indices)
      exit(1)

  print("sem to sig and back test passed.")

  for order in dim_orders:

    test_shape.dim_order = order
    old_coeffs = test_shape.old_get_layout_addressing_coeffs()
    new_coeffs = test_shape.get_layout_addressing_coeffs()
    if old_coeffs != new_coeffs:
      print("new coeffs don't match old coeffs with dim_order "
            "%s" % order)
      print("old coeffs %s\nnew coeffs %s" % (old_coeffs,
                                              new_coeffs))
      #exit(1)

if __name__ == "__main__":
  main()
