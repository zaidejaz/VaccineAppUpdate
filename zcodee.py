import numpy as np
import pandas as pd
from libreco.data import random_split, DatasetPure
from libreco.algorithms import NCF  # pure data,
from libreco.evaluation import evaluate
data = pd.read_csv("ratings.csv")
data.columns = ["user", "item", "label", "time"]
train_data, eval_data, test_data = random_split(data, multi_ratios=[0.8, 0.1, 0.1])
train_data, data_info= DatasetPure.build_trainset(train_data)
eval_data = DatasetPure.build_evalset(eval_data)
test_data = DatasetPure.build_testset(test_data)
ncf = NCF(
    task="rating",
    data_info=data_info,
    loss_type="cross_entropy",
    embed_size=16,
    n_epochs=10,
    lr=1e-3,
    batch_size=2048,
    num_neg=1,
)
# monitor metrics on eval data during training
ncf.fit(
    train_data,
    neg_sampling=False, #for rating, this param is false else True
    verbose=2,
    eval_data=eval_data,
    metrics=["loss"],
)

# do final evaluation on test data
evaluate(
    model=ncf,
    data=test_data,
    neg_sampling=False,
    metrics=["loss"],
)
#for implicit feedback, metrics like precision@k, recall@k, ndcg can be used
# predict preference of user 5755 to item 110
ncf.predict(user=5755, item=110)

# recommend 10items for user 5755
ncf.recommend_user(user=5755, n_rec=10)