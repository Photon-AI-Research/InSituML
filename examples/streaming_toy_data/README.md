* Toy8-INN2018.ipynb: seems to be an inplementation of the INN (not cINN) of
  Ardizzone et al. "Analyzing Inverse Problems with Invertible Neural
  Networks", 2019, http://arxiv.org/abs/1808.04730

* Toy8-cINN.ipynb: cINN training based on Ardizzone et al. "Guided Image
  Generation with Conditional Invertible Neural Networks", 2019,
  http://arxiv.org/abs/1907.02392

  Imports tooling from distributed_toy8_inn2019.py which doesn't exist.

* Toy8-cINN-ContinualLearning.ipynb: Same as Toy8-cINN.ipynb but with modified
  toy data, modeling time-dependent changing condition (= label)

* toy8.py, train_cINN_distributed_toy8.py: tooling used in Toy8-cINN-ContinualLearning.ipynb
