"""
Implements a K-Nearest Neighbor classifier in PyTorch.
"""
import torch
from typing import Dict, List

from numpy.ma.core import reshape


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from knn.py!")


def compute_distances_two_loops(x_train: torch.Tensor, x_test: torch.Tensor):
    """
    Computes the squared Euclidean distance between each element of training
    set and each element of test set. Images should be flattened and treated
    as vectors.

    This implementation uses a naive set of nested loops over the training and
    test data.

    The input data may have any number of dimensions -- for example this
    function should be able to compute nearest neighbor between vectors, in
    which case the inputs will have shape (num_{train, test}, D); it should
    also be able to compute nearest neighbors between images, where the inputs
    will have shape (num_{train, test}, C, H, W). More generally, the inputs
    will have shape (num_{train, test}, D1, D2, ..., Dn); you should flatten
    each element of shape (D1, D2, ..., Dn) into a vector of shape
    (D1 * D2 * ... * Dn) before computing distances.

    The input tensors should not be modified.

    NOTE: Your implementation may not use `torch.norm`, `torch.dist`,
    `torch.cdist`, or their instance method variants (`x.norm`, `x.dist`,
    `x.cdist`, etc.). You may not use any functions from `torch.nn` or
    `torch.nn.functional` modules.

    Args:
        x_train: Tensor of shape (num_train, D1, D2, ...)
        x_test: Tensor of shape (num_test, D1, D2, ...)

    Returns:
        dists: Tensor of shape (num_train, num_test) where dists[i, j]
            is the squared Euclidean distance between the i-th training point
            and the j-th test point. It should have the same dtype as x_train.
    
    计算训练集中每个元素与测试集中每个元素之间的欧几里得距离的平方。
    图像应当被展平并作为向量处理。

    此实现使用了一组简单的嵌套循环来遍历训练数据和测试数据。

    输入数据可以具有任意数量的维度 -- 例如，该函数应该能够计算向量之间的最近邻，
    这种情况下输入的形状将是 (num_{train, test}, D)；它也应该能够计算图像之间的最近邻，
    此时输入的形状将是 (num_{train, test}, C, H, W)。更一般地，输入将具有形状
    (num_{train, test}, D1, D2, ..., Dn)；在计算距离之前，您应该将形状为
    (D1, D2, ..., Dn) 的每个元素展平成形状为 (D1 * D2 * ... * Dn) 的向量。

    输入张量不应被修改。

    注意：您的实现不能使用 `torch.norm`、`torch.dist`、`torch.cdist` 或它们的实例方法
    变体（`x.norm`、`x.dist`、`x.cdist` 等）。您不能使用来自 `torch.nn` 或
    `torch.nn.functional` 模块的任何函数。

    参数：
        x_train: 形状为 (num_train, D1, D2, ...) 的张量
        x_test: 形状为 (num_test, D1, D2, ...) 的张量

    返回：
        dists: 形状为 (num_train, num_test) 的张量，其中 dists[i, j] 是
            第i个训练点和第j个测试点之间的欧几里得距离的平方。它应该与 x_train 具有相同的数据类型。
    
    """

    # Initialize dists to be a tensor of shape (num_train, num_test) with the
    # same datatype and device as x_train
    # 初始化dists为形状为(num_train, num_test)的张量，
    # 与x_train具有相同的数据类型和设备
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    dists = x_train.new_zeros(num_train, num_test)
    ##########################################################################
    # TODO: Implement this function using a pair of nested loops over the    #
    # training data and the test data.                                       #
    #                                                                        #
    # You may not use torch.norm (or its instance method variant), nor any   #
    # functions from torch.nn or torch.nn.functional.                        #
    ##########################################################################
    # Replace "pass" statement with your code
    for i in range(num_train):
        for j in range(num_test):
            dists[i, j] = (x_train[i].reshape(-1) - x_test[j].reshape(-1)).pow(2).sum().sqrt()

    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################
    return dists


def compute_distances_one_loop(x_train: torch.Tensor, x_test: torch.Tensor):
    """
    Computes the squared Euclidean distance between each element of training
    set and each element of test set. Images should be flattened and treated
    as vectors.

    This implementation uses only a single loop over the training data.

    Similar to `compute_distances_two_loops`, this should be able to handle
    inputs with any number of dimensions. The inputs should not be modified.

    NOTE: Your implementation may not use `torch.norm`, `torch.dist`,
    `torch.cdist`, or their instance method variants (`x.norm`, `x.dist`,
    `x.cdist`, etc.). You may not use any functions from `torch.nn` or
    `torch.nn.functional` modules.

    Args:
        x_train: Tensor of shape (num_train, D1, D2, ...)
        x_test: Tensor of shape (num_test, D1, D2, ...)

    Returns:
        dists: Tensor of shape (num_train, num_test) where dists[i, j]
            is the squared Euclidean distance between the i-th training point
            and the j-th test point. It should have the same dtype as x_train.
    """
    # 计算训练集中每个元素与测试集中每个元素之间的欧几里得距离的平方。
    # 图像应当被展平并作为向量处理。
    #
    # 此实现仅使用一个循环来遍历训练数据。
    #
    # 与 `compute_distances_two_loops` 类似，此函数应该能够处理任意维度的输入。
    # 输入不应被修改。
    #
    # 注意：您的实现不能使用 `torch.norm`、`torch.dist`、`torch.cdist` 或它们的实例方法
    # 变体（`x.norm`、`x.dist`、`x.cdist` 等）。您不能使用来自 `torch.nn` 或
    # `torch.nn.functional` 模块的任何函数。
    #
    # 参数：
    #     x_train: 形状为 (num_train, D1, D2, ...) 的张量
    #     x_test: 形状为 (num_test, D1, D2, ...) 的张量
    #
    # 返回：
    #     dists: 形状为 (num_train, num_test) 的张量，其中 dists[i, j] 是
    #         第i个训练点和第j个测试点之间的欧几里得距离的平方。它应该与 x_train 具有相同的数据类型。

    # Initialize dists to be a tensor of shape (num_train, num_test) with the
    # same datatype and device as x_train
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    dists = x_train.new_zeros(num_train, num_test)
    ##########################################################################
    # TODO: Implement this function using only a single loop over x_train.   #
    #                                                                        #
    # You may not use torch.norm (or its instance method variant), nor any   #
    # functions from torch.nn or torch.nn.functional.                        #
    ##########################################################################
    # Replace "pass" statement with your code
    for i in range(num_train):
        dists[i, :] = (x_train[i].reshape(-1) - x_test.reshape(num_test, -1)).pow(2).sum(dim=1).sqrt()
    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################
    return dists


def compute_distances_no_loops(x_train: torch.Tensor, x_test: torch.Tensor):
    """
    Computes the squared Euclidean distance between each element of training
    set and each element of test set. Images should be flattened and treated
    as vectors.

    This implementation should not use any Python loops. For memory-efficiency,
    it also should not create any large intermediate tensors; in particular you
    should not create any intermediate tensors with O(num_train * num_test)
    elements.

    Similar to `compute_distances_two_loops`, this should be able to handle
    inputs with any number of dimensions. The inputs should not be modified.

    NOTE: Your implementation may not use `torch.norm`, `torch.dist`,
    `torch.cdist`, or their instance method variants (`x.norm`, `x.dist`,
    `x.cdist`, etc.). You may not use any functions from `torch.nn` or
    `torch.nn.functional` modules.

    Args:
        x_train: Tensor of shape (num_train, C, H, W)
        x_test: Tensor of shape (num_test, C, H, W)

    Returns:
        dists: Tensor of shape (num_train, num_test) where dists[i, j] is
            the squared Euclidean distance between the i-th training point and
            the j-th test point.

    """
    # 计算训练集中每个元素与测试集中每个元素之间的欧几里得距离的平方。
    # 图像应当被展平并作为向量处理。
    #
    # 此实现不应使用任何Python循环。为了内存效率，
    # 它也不应创建任何大的中间张量；特别是不应创建任何具有
    # O(num_train * num_test)元素的中间张量。
    #
    # 与 `compute_distances_two_loops` 类似，此函数应该能够处理
    # 任意维度的输入。输入不应被修改。
    #
    # 注意：您的实现不能使用 `torch.norm`、`torch.dist`、`torch.cdist`
    # 或它们的实例方法变体（`x.norm`、`x.dist`、`x.cdist` 等）。
    # 您不能使用来自 `torch.nn` 或 `torch.nn.functional` 模块的任何函数。
    #
    # 参数：
    #     x_train: 形状为 (num_train, C, H, W) 的张量
    #     x_test: 形状为 (num_test, C, H, W) 的张量
    #
    # 返回：
    #     dists: 形状为 (num_train, num_test) 的张量，其中 dists[i, j]
    #         是第i个训练点和第j个测试点之间的欧几里得距离的平方。

    # Initialize dists to be a tensor of shape (num_train, num_test) with the
    # same datatype and device as x_train
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    dists = x_train.new_zeros(num_train, num_test)
    ##########################################################################
    # TODO: Implement this function without using any explicit loops and     #
    # without creating any intermediate tensors with O(num_train * num_test) #
    # elements.                                                              #
    #                                                                        #
    # You may not use torch.norm (or its instance method variant), nor any   #
    # functions from torch.nn or torch.nn.functional.                        #
    #                                                                        #
    # HINT: Try to formulate the Euclidean distance using two broadcast sums #
    #       and a matrix multiply.                                           #
    # TODO: 实现这个函数，不使用任何显式循环，且
    # 不创建任何具有 O(num_train * num_test) 元素的中间张量。
    #
    # 您不能使用 torch.norm（或其实例方法变体），也不能使用
    # 来自 torch.nn 或 torch.nn.functional 的任何函数。
    #
    # 提示：尝试使用两个广播求和和一个矩阵乘法来表示欧几里得距离。
    ##########################################################################
    # Replace "pass" statement with your code

    # 这个实现可以完成任务 但是很慢
    # train = x_train.reshape(num_train,1, -1) #(100,1,768)
    # test = x_test.reshape(1, num_test,-1) #(1,100,768)
    # dists = (train - test).pow(2).sum(dim=2).sqrt()

    # 对于向量x,y 那么向量之间的距离是sqrt((x-y)*(x-y)) = sqrt(x*x - 2x*y + y*y)
    A = x_train.reshape(num_train, -1)
    B = x_test.reshape(num_test, -1)
    AB2 = A.mm(B.T) * 2
    dists = ((A ** 2).sum(dim=1).reshape(-1, 1) - AB2 + (B ** 2).sum(dim=1).reshape(1, -1)) ** (1 / 2)

    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################
    return dists


def predict_labels(dists: torch.Tensor, y_train: torch.Tensor, k: int = 1):
    """
    Given distances between all pairs of training and test samples, predict a
    label for each test sample by taking a MAJORITY VOTE among its `k` nearest
    neighbors in the training set.

    In the event of a tie, this function SHOULD return the smallest label. For
    example, if k=5 and the 5 nearest neighbors to a test example have labels
    [1, 2, 1, 2, 3] then there is a tie between 1 and 2 (each have 2 votes),
    so we should return 1 since it is the smallest label.

    This function should not modify any of its inputs.

    Args:
        dists: Tensor of shape (num_train, num_test) where dists[i, j] is the
            squared Euclidean distance between the i-th training point and the
            j-th test point.
        y_train: Tensor of shape (num_train,) giving labels for all training
            samples. Each label is an integer in the range [0, num_classes - 1]
        k: The number of nearest neighbors to use for classification.

    Returns:
        y_pred: int64 Tensor of shape (num_test,) giving predicted labels for
            the test data, where y_pred[j] is the predicted label for the j-th
            test example. Each label should be an integer in the range
            [0, num_classes - 1].

    给定所有训练样本和测试样本之间的距离，通过对每个测试样本的 `k` 个最近邻进行多数投票来预测其标签。

    如果出现平局，此函数应返回最小的标签。例如，如果 k=5 且某个测试样本的 5 个最近邻的标签为 [1, 2, 1, 2, 3]，则 1 和 2 之间出现平局（各有 2 票），因此我们应该返回 1，因为它是较小的标签。

    此函数不应修改其任何输入。

    参数：
        dists: 形状为 (num_train, num_test) 的张量，其中 dists[i, j] 是第 i 个训练样本和第 j 个测试样本之间的平方欧几里得距离。
        y_train: 形状为 (num_train,) 的张量，给出所有训练样本的标签。每个标签是范围 [0, num_classes - 1] 内的整数。
        k: 用于分类的最近邻的数量。

    返回：
        y_pred: 形状为 (num_test,) 的 int64 张量，给出测试数据的预测标签，其中 y_pred[j] 是第 j 个测试样本的预测标签。每个标签应为范围 [0, num_classes - 1] 内的整数。
    """
    num_train, num_test = dists.shape
    y_pred = torch.zeros(num_test, dtype=torch.int64)
    ##########################################################################
    # TODO: Implement this function. You may use an explicit loop over the   #
    # test samples.                                                          #
    #                                                                        #
    # HINT: Look up the function torch.topk                                  #
    ##########################################################################
    # Replace "pass" statement with your code

    def get_min_mode(tensor):
        """
        获取tensor中的最小众数

        Args:
            tensor (torch.Tensor): 输入tensor

        Returns:
            int or float: tensor中出现次数最多的数字中的最小值
        """
        # 将tensor展平为1维
        flat_tensor = tensor.flatten()

        # 获取唯一值及其计数
        unique_values, counts = torch.unique(flat_tensor, return_counts=True)

        # 找到最大计数
        max_count = torch.max(counts)

        # 获取所有具有最大计数的值
        mode_mask = counts == max_count
        modes = unique_values[mode_mask]

        # 返回众数中的最小值
        return torch.min(modes).item()






    for i in range(num_test):
        indices = torch.topk(dists[:,i].reshape(-1),k,largest=False).indices
        label = y_train[indices]
        # y_pred[i] = label.mode().values.item() #仅在cpu上的实现为返回最小的众数
        y_pred[i] = get_min_mode(label)
        ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################
    return y_pred


class KnnClassifier:

    def __init__(self, x_train: torch.Tensor, y_train: torch.Tensor):
        """
        Create a new K-Nearest Neighbor classifier with the specified training
        data. In the initializer we simply memorize the provided training data.

        Args:
            x_train: Tensor of shape (num_train, C, H, W) giving training data
            y_train: int64 Tensor of shape (num_train, ) giving training labels

        使用分类器进行预测。

        参数:
          x_test: 形状为 (num_test, C, H, W) 的张量，提供测试样本。
          k: 用于预测的邻居数量。

        返回:
          y_test_pred: 形状为 (num_test,) 的张量，提供测试样本的预测标签。
        """
        ######################################################################
        # TODO: Implement the initializer for this class. It should perform  #
        # no computation and simply memorize the training data in            #
        # `self.x_train` and `self.y_train`, accordingly.                    #
        # 为这个类实现初始化器。它不应该执行任何计算，只需将训练数据分别记住在 
        # `self.x_train` 和 `self.y_train` 中。                             
        ######################################################################
        # Replace "pass" statement with your code
        pass
        ######################################################################
        #                         END OF YOUR CODE                           #
        ######################################################################

    def predict(self, x_test: torch.Tensor, k: int = 1):
        """
        Make predictions using the classifier.

        Args:
            x_test: Tensor of shape (num_test, C, H, W) giving test samples.
            k: The number of neighbors to use for predictions.

        Returns:
            y_test_pred: Tensor of shape (num_test,) giving predicted labels
                for the test samples.
        使用分类器进行预测。

        参数:
          x_test: 形状为 (num_test, C, H, W) 的张量，提供测试样本。
          k: 用于预测的邻居数量。

        返回:
          y_test_pred: 形状为 (num_test,) 的张量，提供测试样本的预测标签。
        """
        y_test_pred = None
        ######################################################################
        # TODO: Implement this method. You should use the functions you      #
        # wrote above for computing distances (use the no-loop variant) and  #
        # to predict output labels.                                          #
        ######################################################################
        # Replace "pass" statement with your code
        pass
        ######################################################################
        #                         END OF YOUR CODE                           #
        ######################################################################
        return y_test_pred

    def check_accuracy(
            self,
            x_test: torch.Tensor,
            y_test: torch.Tensor,
            k: int = 1,
            quiet: bool = False
    ):
        """
        Utility method for checking the accuracy of this classifier on test
        data. Returns the accuracy of the classifier on the test data, and
        also prints a message giving the accuracy.

        Args:
            x_test: Tensor of shape (num_test, C, H, W) giving test samples.
            y_test: int64 Tensor of shape (num_test,) giving test labels.
            k: The number of neighbors to use for prediction.
            quiet: If True, don't print a message.

        Returns:
            accuracy: Accuracy of this classifier on the test data, as a
                percent. Python float in the range [0, 100]
        """
        y_test_pred = self.predict(x_test, k=k)
        num_samples = x_test.shape[0]
        num_correct = (y_test == y_test_pred).sum().item()
        accuracy = 100.0 * num_correct / num_samples
        msg = (
            f"Got {num_correct} / {num_samples} correct; "
            f"accuracy is {accuracy:.2f}%"
        )
        if not quiet:
            print(msg)
        return accuracy


def knn_cross_validate(
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        num_folds: int = 5,
        k_choices: List[int] = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100],
):
    """
    Perform cross-validation for `KnnClassifier`.

    Args:
        x_train: Tensor of shape (num_train, C, H, W) giving all training data.
        y_train: int64 Tensor of shape (num_train,) giving labels for training
            data.
        num_folds: Integer giving the number of folds to use.
        k_choices: List of integers giving the values of k to try.

    Returns:
        k_to_accuracies: Dictionary mapping values of k to lists, where
            k_to_accuracies[k][i] is the accuracy on the i-th fold of a
            `KnnClassifier` that uses k nearest neighbors.
    """

    # First we divide the training data into num_folds equally-sized folds.
    x_train_folds = []
    y_train_folds = []
    ##########################################################################
    # TODO: Split the training data and images into folds. After splitting,  #
    # x_train_folds and y_train_folds should be lists of length num_folds,   #
    # where y_train_folds[i] is label vector for images inx_train_folds[i].  #
    #                                                                        #
    # HINT: torch.chunk                                                      #
    ##########################################################################
    # Replace "pass" statement with your code
    pass
    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################

    # A dictionary holding the accuracies for different values of k that we
    # find when running cross-validation. After running cross-validation,
    # k_to_accuracies[k] should be a list of length num_folds giving the
    # different accuracies we found trying `KnnClassifier`s using k neighbors.
    k_to_accuracies = {}

    ##########################################################################
    # TODO: Perform cross-validation to find the best value of k. For each   #
    # value of k in k_choices, run the k-NN algorithm `num_folds` times; in  #
    # each case you'll use all but one fold as training data, and use the    #
    # last fold as a validation set. Store the accuracies for all folds and  #
    # all values in k in k_to_accuracies.                                    #
    #                                                                        #
    # HINT: torch.cat                                                        #
    ##########################################################################
    # Replace "pass" statement with your code
    pass
    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################

    return k_to_accuracies


def knn_get_best_k(k_to_accuracies: Dict[int, List]):
    """
    Select the best value for k, from the cross-validation result from
    knn_cross_validate. If there are multiple k's available, then you SHOULD
    choose the smallest k among all possible answer.

    Args:
        k_to_accuracies: Dictionary mapping values of k to lists, where
            k_to_accuracies[k][i] is the accuracy on the i-th fold of a
            `KnnClassifier` that uses k nearest neighbors.

    Returns:
        best_k: best (and smallest if there is a conflict) k value based on
            the k_to_accuracies info.
    """
    best_k = 0
    ##########################################################################
    # TODO: Use the results of cross-validation stored in k_to_accuracies to #
    # choose the value of k, and store result in `best_k`. You should choose #
    # the value of k that has the highest mean accuracy accross all folds.   #
    ##########################################################################
    # Replace "pass" statement with your code
    pass
    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################
    return best_k
