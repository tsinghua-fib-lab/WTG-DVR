# -*- coding:utf-8 -*-

from numpy.core.fromnumeric import mean
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.python.keras.layers import Flatten

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import os


class NoMask(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NoMask, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(NoMask, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):
        return x

    def compute_mask(self, inputs, mask):
        return None


class Hash(tf.keras.layers.Layer):
    """
    hash the input to [0,num_buckets)
    if mask_zero = True,0 or 0.0 will be set to 0,other value will be set in range[1,num_buckets)
    """

    def __init__(self, num_buckets, mask_zero=False, **kwargs):
        self.num_buckets = num_buckets
        self.mask_zero = mask_zero
        super(Hash, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(Hash, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):


        if x.dtype != tf.string:
            zero = tf.as_string(tf.zeros([1], dtype=x.dtype))
            x = tf.as_string(x, )
        else:
            zero = tf.as_string(tf.zeros([1], dtype='int32'))

        num_buckets = self.num_buckets if not self.mask_zero else self.num_buckets - 1
        try:
            hash_x = tf.string_to_hash_bucket_fast(x, num_buckets,
                                                   name=None)  # weak hash
        except:
            hash_x = tf.strings.to_hash_bucket_fast(x, num_buckets,
                                                    name=None)  # weak hash
        if self.mask_zero:
            mask = tf.cast(tf.not_equal(x, zero), dtype='int64')
            hash_x = (hash_x + 1) * mask

        return hash_x
    def get_config(self, ):
        config = {'num_buckets': self.num_buckets, 'mask_zero': self.mask_zero, }
        base_config = super(Hash, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Linear(tf.keras.layers.Layer):

    def __init__(self, l2_reg=0.0, mode=0, use_bias=False, seed=1024, **kwargs):

        self.l2_reg = l2_reg
        # self.l2_reg = tf.contrib.layers.l2_regularizer(float(l2_reg_linear))
        if mode not in [0, 1, 2]:
            raise ValueError("mode must be 0,1 or 2")
        self.mode = mode
        self.use_bias = use_bias
        self.seed = seed
        super(Linear, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.use_bias:
            self.bias = self.add_weight(name='linear_bias',
                                        shape=(1,),
                                        initializer=tf.keras.initializers.Zeros(),
                                        trainable=True)
        if self.mode == 1:
            self.kernel = self.add_weight(
                'linear_kernel',
                shape=[int(input_shape[-1]), 1],
                initializer=tf.keras.initializers.glorot_normal(self.seed),
                regularizer=tf.keras.regularizers.l2(self.l2_reg),
                trainable=True)
        elif self.mode == 2:
            self.kernel = self.add_weight(
                'linear_kernel',
                shape=[int(input_shape[1][-1]), 1],
                initializer=tf.keras.initializers.glorot_normal(self.seed),
                regularizer=tf.keras.regularizers.l2(self.l2_reg),
                trainable=True)

        super(Linear, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        if self.mode == 0:
            sparse_input = inputs
            linear_logit = reduce_sum(sparse_input, axis=-1, keep_dims=True)
        elif self.mode == 1:
            dense_input = inputs
            fc = tf.tensordot(dense_input, self.kernel, axes=(-1, 0))
            linear_logit = fc
        else:
            sparse_input, dense_input = inputs
            fc = tf.tensordot(dense_input, self.kernel, axes=(-1, 0))
            linear_logit = reduce_sum(sparse_input, axis=-1, keep_dims=False) + fc
        if self.use_bias:
            linear_logit += self.bias

        return linear_logit

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def compute_mask(self, inputs, mask):
        return None

    def get_config(self, ):
        config = {'mode': self.mode, 'l2_reg': self.l2_reg, 'use_bias': self.use_bias, 'seed': self.seed}
        base_config = super(Linear, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def concat_func(inputs, axis=-1, mask=False):
    if not mask:
        inputs = list(map(NoMask(), inputs))
    if len(inputs) == 1:
        return inputs[0]
    else:
        return tf.keras.layers.Concatenate(axis=axis)(inputs)


def reduce_mean(input_tensor,
                axis=None,
                keep_dims=False,
                name=None,
                reduction_indices=None):
    try:
        return tf.reduce_mean(input_tensor,
                              axis=axis,
                              keep_dims=keep_dims,
                              name=name,
                              reduction_indices=reduction_indices)
    except TypeError:
        return tf.reduce_mean(input_tensor,
                              axis=axis,
                              keepdims=keep_dims,
                              name=name)


def reduce_sum(input_tensor,
               axis=None,
               keep_dims=False,
               name=None,
               reduction_indices=None):
    try:
        return tf.reduce_sum(input_tensor,
                             axis=axis,
                             keep_dims=keep_dims,
                             name=name,
                             reduction_indices=reduction_indices)
    except TypeError:
        return tf.reduce_sum(input_tensor,
                             axis=axis,
                             keepdims=keep_dims,
                             name=name)


def reduce_max(input_tensor,
               axis=None,
               keep_dims=False,
               name=None,
               reduction_indices=None):
    try:
        return tf.reduce_max(input_tensor,
                             axis=axis,
                             keep_dims=keep_dims,
                             name=name,
                             reduction_indices=reduction_indices)
    except TypeError:
        return tf.reduce_max(input_tensor,
                             axis=axis,
                             keepdims=keep_dims,
                             name=name)


def div(x, y, name=None):
    try:
        return tf.div(x, y, name=name)
    except AttributeError:
        return tf.divide(x, y, name=name)


def softmax(logits, dim=-1, name=None):
    try:
        return tf.nn.softmax(logits, dim=dim, name=name)
    except TypeError:
        return tf.nn.softmax(logits, axis=dim, name=name)


class Add(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Add, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(Add, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if not isinstance(inputs, list):
            return inputs
        if len(inputs) == 1:
            return inputs[0]
        if len(inputs) == 0:
            return tf.constant([[0.0]])

        return tf.keras.layers.add(inputs)


def add_func(inputs):
    return Add()(inputs)


def combined_dnn_input(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = Flatten()(concat_func(sparse_embedding_list))
        dense_dnn_input = Flatten()(concat_func(dense_value_list))
        return concat_func([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return Flatten()(concat_func(sparse_embedding_list))
    elif len(dense_value_list) > 0:
        return Flatten()(concat_func(dense_value_list))
    else:
        raise NotImplementedError("dnn_feature_columns can not be empty list")


def metrics_at_k(y_true, watchtime_true, gain_true, like_true, y_pred, users, duration, topk, round_position, save_result, save_result_path, save_result_name, dataset, model, train_target, disentangle, iid, bc_thres):

    if save_result:
        os.makedirs(save_result_path, exist_ok=True)

    df = pd.DataFrame({
        "uid": users, 
        "duration": duration, 
        "y_true": y_true, 
        "y_pred": y_pred, 
        "watchtime_true": watchtime_true,
        "gain_true": gain_true,
        "like_true": like_true,
        "iid": iid})

    if save_result:
        filename = os.path.join(save_result_path, 'result-all-{}.csv'.format(save_result_name))
        df.to_csv(filename)

    valid_users = df[['uid', 'duration']].groupby('uid').count().reset_index().rename(columns={'duration': 'count'})
    valid_users = valid_users[valid_users['count'] > topk]
    df = df.merge(valid_users, on='uid')

    df['rank'] = df['y_pred'].groupby(df['uid']).rank(method='first', ascending=False)

    df_topk = df[df['rank'] <= topk].copy()

    if save_result:
        filename = os.path.join(save_result_path, 'result-topk-{}.csv'.format(save_result_name))
        df_topk.to_csv(filename)

    mae_at_k = df_topk[['uid', 'y_true', 'y_pred']].groupby('uid').apply(groupby_mae)
    mean_mae_at_k = round(mae_at_k.mean(), round_position)
    rmse_at_k = df_topk[['uid', 'y_true', 'y_pred']].groupby('uid').apply(groupby_rmse)
    mean_rmse_at_k = round(rmse_at_k.mean(), round_position)
    watchtime_at_k = df_topk[['uid', 'watchtime_true']].groupby('uid').sum()
    mean_watchtime_at_k = round(watchtime_at_k['watchtime_true'].mean(), round_position)
    gain_at_k = df_topk[['uid', 'gain_true']].groupby('uid').mean()
    mean_gain_at_k = round(gain_at_k['gain_true'].mean(), round_position)
    df_topk['dc_gain'] = df_topk['gain_true']*(1/np.log(1 + df_topk['rank']))
    dcg_at_k = df_topk[['uid', 'dc_gain']].groupby('uid').sum()
    mean_dcg_at_k = round(dcg_at_k['dc_gain'].mean(), round_position)
    sum_like_at_k = df_topk['like_true'].sum()
    sum_bc_at_k = (df_topk['watchtime_true'] <= bc_thres).sum()

    return mean_mae_at_k, mean_rmse_at_k, mean_watchtime_at_k, mean_gain_at_k, mean_dcg_at_k, sum_like_at_k, sum_bc_at_k


def metrics_at_t(y_true, watchtime_true, gain_true, like_true, y_pred, users, duration, topt, round_position):

    df = pd.DataFrame({
        "uid": users, 
        "duration": duration, 
        "y_true": y_true, 
        "y_pred": y_pred, 
        "watchtime_true": watchtime_true,
        "gain_true": gain_true,
        "like_true": like_true})


    valid_users = df[['uid', 'duration']].groupby('uid').sum().reset_index().rename(columns={'duration': 'sum'})
    valid_users = valid_users[valid_users['sum'] > topt]
    df = df.merge(valid_users, on='uid')

    df = df.sort_values('y_pred', ascending=False)
    df['cumsum_duration'] = df[['uid', 'duration']].groupby('uid').cumsum()
    df['rank'] = df.groupby('uid').cumcount()

    df_below_t = df[df['cumsum_duration'] < topt]
    df_below_t_rank = df_below_t[['uid', 'rank']].groupby('uid').max().reset_index().rename(columns={'rank': 'max_rank'})
    df_below_t_rank['cut_rank'] = df_below_t_rank['max_rank'] + 1

    df = df.merge(df_below_t_rank[['uid', 'cut_rank']], on='uid')
    df_full_video = df[df['rank'] < df['cut_rank']].copy()
    df_cut_video = df[df['rank'] == df['cut_rank']].copy()

    df_cut_video['scale'] = 1 - (df_cut_video['cumsum_duration'] - topt)/df_cut_video['duration']
    df_cut_video['duration'] = df_cut_video['duration']*df_cut_video['scale']
    df_cut_video['y_true'] = df_cut_video['y_true']*df_cut_video['scale']
    df_cut_video['y_pred'] = df_cut_video['y_pred']*df_cut_video['scale']
    df_cut_video['watchtime_true'] = df_cut_video['watchtime_true']*df_cut_video['scale']
    df_cut_video['gain_true'] = df_cut_video['gain_true']*df_cut_video['scale']

    df_topt = pd.concat([df_full_video, df_cut_video])

    mae_at_t = df_topt[['uid', 'y_true', 'y_pred']].groupby('uid').apply(groupby_mae)
    mean_mae_at_t = round(mae_at_t.mean(), round_position)
    rmse_at_t = df_topt[['uid', 'y_true', 'y_pred']].groupby('uid').apply(groupby_rmse)
    mean_rmse_at_t = round(rmse_at_t.mean(), round_position)
    watchtime_at_t = df_topt[['uid', 'watchtime_true']].groupby('uid').sum()
    mean_watchtime_at_t = round(watchtime_at_t['watchtime_true'].mean(), round_position)
    gain_at_t = df_topt[['uid', 'gain_true']].groupby('uid').sum()
    mean_gain_at_t = round(gain_at_t['gain_true'].mean(), round_position)
    df_topt['area_gain'] = df_topt['gain_true']*(df_topt['cut_rank'] + 1 - df_topt['rank'])
    auc_gain_at_t = df_topt[['uid', 'area_gain']].groupby('uid').sum()
    mean_auc_gain_at_t = round(auc_gain_at_t['area_gain'].mean(), round_position)
    sum_like_at_t = df_topt['like_true'].sum()

    return mean_mae_at_t, mean_rmse_at_t, mean_watchtime_at_t, mean_gain_at_t, mean_auc_gain_at_t, sum_like_at_t


def groupby_mae(df):

    y_hat = df.y_pred
    y = df.y_true

    return mean_absolute_error(y, y_hat)


def groupby_rmse(df):

    y_hat = df.y_pred
    y = df.y_true

    return mean_squared_error(y, y_hat, squared=False)


def gauc_score(users, y_true, y_pred):

    df = pd.DataFrame({"users": users, "y_true": y_true, "y_pred": y_pred})
    all_true_users = df[["users", "y_true"]].groupby("users").all().reset_index().rename(columns={"y_true": "all_true"})
    all_false_users = df[["users", "y_true"]].groupby("users").any().reset_index().rename(columns={"y_true": "all_false"})
    valid_users = all_true_users.merge(all_false_users, on="users")
    valid_users = valid_users[(valid_users["all_true"] == False) & (valid_users["all_false"] == True)]
    df = df.merge(valid_users["users"], on="users")
    weight = df[["users", "y_true"]].groupby("users").count().reset_index().set_index("users", drop=True).rename(columns={"y_true": "weight"})
    weight["weight"] = weight["weight"]/weight["weight"].sum()

    weight["auc"] = df.groupby("users").apply(groupby_auc)
    gauc_score = (weight["weight"]*weight["auc"]).sum()
    weight.drop(columns="auc", inplace=True)

    return gauc_score


def groupby_auc(df):

    y_hat = df.y_pred
    y = df.y_true

    return roc_auc_score(y, y_hat)
