# -*- coding:utf-8 -*-

from itertools import chain

import tensorflow as tf

from ..feature_column import build_input_features, get_linear_logit, DEFAULT_GROUP_NAME, input_from_feature_columns
from ..layers.core import PredictionLayer, DNN
from ..layers.interaction import FM
from ..layers.utils import concat_func, add_func, combined_dnn_input


def LibFM(features, linear_feature_columns, dnn_feature_columns, fm_group=[DEFAULT_GROUP_NAME], 
           l2_reg_linear=0.00001, l2_reg_embedding=0.00001, seed=1024, task='binary', output_name='prediction'):
    """Instantiates the LibFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param fm_group: list, group_name of features that will be used to do feature interactions.
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear', suffix=output_name, l2_reg=l2_reg_linear)

    group_embedding_dict, _ = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding, seed, support_group=True, suffix=output_name)

    fm_logit = add_func([FM()(concat_func(v, axis=1)) for k, v in group_embedding_dict.items() if k in fm_group])

    final_logit = add_func([linear_logit, fm_logit])

    output = PredictionLayer(task, name=output_name)(final_logit)
    return output
