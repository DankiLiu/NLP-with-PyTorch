from pyparsing import oneOf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import seaborn as sns
import matplotlib.pyplot as plt

import torch
import numpy as np

corpus = ['Time flies like an arrow.',
          'Fruit flies like a banana.']


def example_1_1(corpus):
    """One-hot encoding using scikit-learn. (Page 7)"""
    one_hot_vectorizer = CountVectorizer(binary=True)
    one_hot = one_hot_vectorizer.fit_transform(corpus).toarray()
    print("one_hot: ", one_hot)
    sns.heatmap(one_hot, annot=True,
                cbar=False, xticklabels=one_hot_vectorizer.get_feature_names(),
                yticklabels=['Sentence 1', 'Sentence 2'])
    plt.show()


def example_1_2(corpus):
    """TF-IDF representation using scikit-learn. (P.9)"""
    tfidf_vectorizer = TfidfVectorizer()
    tfidf = tfidf_vectorizer.fit_transform(corpus).toarray()
    sns.heatmap(tfidf, annot=True,
                cbar=False, xticklabels=tfidf_vectorizer.get_feature_names(),
                yticklabels=['Sentence 1', 'Sentence 2'])


def describe(x):
    print("Type: {}".format(x.type()))
    print("Shape/Size: {}".format(x.shape))
    print("Values: \n{}".format(x))


def example_1_3():
    """Create a Tensor."""
    print("Create a Tensor.")
    # describe(torch.Tensor(2, 3))
    """Create Tensor with random numbers."""
    print("Create Tensor with random numbers.")
    describe(torch.rand(2, 3))
    describe(torch.randn(2, 3))
    """Create a Tensor with zeros and change the its value."""
    print("Create a Tensor with zeros and change the its value.")
    describe(torch.zeros(2, 3))
    x = torch.ones(2, 3)
    x.fill_(5)
    describe(x)
    """Create a Tensor with the help of Python list."""
    print("Create a Tensor with the help of Python list.")
    x = torch.Tensor([[1, 2, 3],
                      [4, 5, 6]])
    describe(x)
    """Create a Tensor using a NumPy array."""
    print("Create a Tensor using a NumPy array.")
    np_array = np.random.rand(2, 3)
    describe(torch.from_numpy(np_array))


def example_1_10():
    """Create a 1-D Tensor with (start, end, step)."""
    x = torch.arange(2, 6)
    describe(x)
    """Returns a new tensor with the same data as the self tensor but of a different shape."""
    x = x.view(2, 2)
    describe(x)
    """Sum each dimension."""
    describe(torch.sum(x, dim=0))
    describe(torch.sum(x, dim=1))
    """Transpose a Tensor."""
    describe(torch.transpose(x, 0, 1))
    y = torch.arange(1, 9)
    y = y.view(2, 2, 2)
    describe(torch.transpose(y, 0, 1))


def example_1_11():
    """Slice and index a Tensor."""
    x = torch.arange(6).view(2, 3)
    describe(x)
    describe(x[:1, :2])
    describe(x[0, 2])


def example_1_12():
    """Complex index of a Tensor."""
    x = torch.Tensor([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
    describe(x)
    indices = torch.LongTensor([0, 1])
    describe(torch.index_select(x, dim=0, index=indices))

    """???"""
    row_indices = torch.arange(2).long()
    col_indices = torch.LongTensor([0, 1])
    indexed = torch.index_select(x, dim=0, index=row_indices)
    describe(indexed)
    describe(torch.index_select(indexed, dim=1, index=col_indices))
    describe(x[row_indices, col_indices])


def example_1_13():
    """Concatenate Tensors."""
    x = torch.arange(6).view(2, 3)
    describe(x)
    describe(torch.cat([x, x], dim=0))
    describe(torch.cat([x, x], dim=1))
    describe(torch.stack([x, x]))


def example_1_14():
    """Linear Algebra with Tensors."""
    x1 = torch.arange(6).view(2, 3).float()
    describe(x1)
    x2 = torch.ones(3, 2)
    x2[:, 1] += 1
    describe(x2)
    describe(torch.mm(x1, x2))


def example_1_15():
    """PyTorch Autograd."""
    x = torch.ones(2, 2, requires_grad=True)
    describe(x)
    print(x.grad is None)
    y = (x + 2) * (x + 5) + 3
    describe(y)
    print(x.grad is None)
    z = y.sum()
    describe(z)
    z.backward()
    print(x.grad)


def example_1_16():
    """GPU and Cuda."""
    print(torch.cuda.is_available())

example_1_14()
