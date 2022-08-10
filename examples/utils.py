import yaml
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from deepctr.feature_column import SparseFeat, DenseFeat, build_input_features
from deepctr.layers.core import PredictionLayer
from deepctr.models import DeepFM, NFM, AFM, LibFM, WDL, AutoInt, AFN
import tensorflow as tf
from tensorflow import keras


def load_yaml(filename):
    """Load a yaml file.

    Args:
        filename (str): Filename.

    Returns:
        dict: Dictionary.
    """
    try:
        with open(filename, "r") as f:
            config = yaml.load(f, yaml.SafeLoader)
        return config
    except FileNotFoundError:  # for file not found
        raise
    except Exception:  # for other exceptions
        raise IOError("load {0} error!".format(filename))


def get_data(flags_obj, cfg):

    file_path = os.path.join('data', cfg['data'][flags_obj.dataset]['file_path'])
    data = pd.read_csv(file_path, index_col=0)
    data = data[cfg['data'][flags_obj.dataset]['used_feature_columns']]

    if flags_obj.dataset == 'wechat':
        data['bgm_song_id'] = data['bgm_song_id'].astype('Int64').astype(str)
        data['bgm_singer_id'] = data['bgm_singer_id'].astype('Int64').astype(str)

    return data


def get_feature_target_duration_mean_std(flags_obj, cfg, data):

    sparse_features = cfg['data'][flags_obj.dataset]['sparse_features']
    dense_features = cfg['data'][flags_obj.dataset]['dense_features']
    watchtime_target = [cfg['data'][flags_obj.dataset]['watchtime_feature_name']]
    gain_target = [cfg['data'][flags_obj.dataset]['gain_feature_name']]
    like = [cfg['data'][flags_obj.dataset]['like_feature_name']]
    duration = [cfg['data'][flags_obj.dataset]['duration_feature_name']]
    mean_play = [cfg['data'][flags_obj.dataset]['mean_play_feature_name']]
    std_play = [cfg['data'][flags_obj.dataset]['std_play_feature_name']]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    if len(dense_features) > 0:
        data[dense_features] = data[dense_features].fillna(0, )

    #Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    if flags_obj.remove_duration_feature:
        dense_features.remove(cfg['data'][flags_obj.dataset]['duration_feature_name'])

    if len(dense_features) > 0:
        fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=flags_obj.embedding_dim)
                            for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                            for feat in dense_features]
    else:
        fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=flags_obj.embedding_dim) for i,feat in enumerate(sparse_features)]

    return data, fixlen_feature_columns, watchtime_target, gain_target, like, duration, mean_play, std_play


def split_data(flags_obj, cfg, data, feature_names):

    train_val_split = cfg['data'][flags_obj.dataset]['train_val_split']
    val_test_split = cfg['data'][flags_obj.dataset]['val_test_split']
    timestamp_key = cfg['data'][flags_obj.dataset]['timestamp_key']
    train = data[data[timestamp_key] < train_val_split]
    val_test = data[data[timestamp_key] >= train_val_split]
    val = val_test[val_test[timestamp_key] < val_test_split]
    test = val_test[val_test[timestamp_key] >= val_test_split]

    train_model_input = {name:train[name].values for name in feature_names}
    val_model_input = {name:val[name].values for name in feature_names}
    test_model_input = {name:test[name].values for name in feature_names}

    return (train, val, test), (train_model_input, val_model_input, test_model_input)


def get_model(flags_obj, cfg, linear_feature_columns, dnn_feature_columns):

    if flags_obj.model == 'Random':
        model = RandomModel()
        return model
    elif flags_obj.model == 'Long':
        model = LongModel()
        return model
    elif flags_obj.model == 'Short':
        model = ShortModel()
        return model
    elif flags_obj.model == 'Optimum':
        model = OptimumModel()
        return model

    features = build_input_features(linear_feature_columns + dnn_feature_columns)
    inputs_list = list(features.values())

    if not flags_obj.disentangle:
        output_name = 'prediction'
    else:
        output_name = 'independent'

    if flags_obj.model == 'DeepFM':
        output = DeepFM(features, linear_feature_columns, dnn_feature_columns, task='regression', 
            dnn_hidden_units=(flags_obj.dnn_hidden_units, flags_obj.dnn_hidden_units), output_name=output_name)
    elif flags_obj.model == 'NFM':
        output = NFM(features, linear_feature_columns, dnn_feature_columns, task='regression',
            dnn_hidden_units=(flags_obj.dnn_hidden_units, flags_obj.dnn_hidden_units), output_name=output_name)
    elif flags_obj.model == 'AFM':
        output = AFM(features, linear_feature_columns, dnn_feature_columns, task='regression', output_name=output_name)
    elif flags_obj.model == 'LibFM':
        output = LibFM(features, linear_feature_columns, dnn_feature_columns, task='regression', output_name=output_name)
    elif flags_obj.model == 'WDL':
        output = WDL(features, linear_feature_columns, dnn_feature_columns, task='regression', 
            dnn_hidden_units=(flags_obj.dnn_hidden_units, flags_obj.dnn_hidden_units), output_name=output_name)
    elif flags_obj.model == 'AutoInt':
        output = AutoInt(features, linear_feature_columns, dnn_feature_columns, task='regression',
            dnn_hidden_units=(flags_obj.dnn_hidden_units, flags_obj.dnn_hidden_units), output_name=output_name)
    elif flags_obj.model == 'AFN':
        output = AFN(features, linear_feature_columns, dnn_feature_columns, task='regression',
            dnn_hidden_units=(flags_obj.dnn_hidden_units, flags_obj.dnn_hidden_units), output_name=output_name)
    else:
        raise ValueError('Model {} not supported!'.format(flags_obj.model))

    if flags_obj.disentangle:
        output = DVR(flags_obj, output)

    model = keras.models.Model(inputs=inputs_list, outputs=output)
    return model


class RandomModel(object):

    def predict(self, duration):
        pred = np.random.permutation(duration)
        return pred


class LongModel(object):

    def predict(self, duration):
        pred = duration
        return pred


class ShortModel(object):

    def predict(self, duration):
        pred = duration.max() + duration.min() - duration
        return pred


class OptimumModel(object):

    def predict(self, label):
        pred = label
        return pred


def DVR(flags_obj, output_independent):

    output = PredictionLayer(task='regression', name='prediction')(output_independent)

    independent_predictor = GradientReversalLayer()(output_independent)
    reconstructor = keras.layers.Dense(1, name='reconstructor')
    independent_reconstruct = reconstructor(independent_predictor)
    independent_reconstruct = PredictionLayer(task='regression', name='independent_reconstruct')(independent_reconstruct)
    return [output, independent_reconstruct]


@tf.custom_gradient
def GradientReversalOperator(x):
    def grad(dy):
        return -1 * dy
    return x, grad


class GradientReversalLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()
        
    def call(self, inputs):
        return GradientReversalOperator(inputs)


def get_callbacks(flags_obj, save_path):

    ckpt_path = os.path.join(save_path, 'ckpt')
    log_path = os.path.join(save_path, 'log')
    tb_path = os.path.join(save_path, 'tb')
    os.makedirs(ckpt_path)
    os.makedirs(log_path)
    os.makedirs(tb_path)

    if not flags_obj.disentangle:
        es = keras.callbacks.EarlyStopping(monitor='val_RMSE', mode='min', patience=3, restore_best_weights=True)
        ckpt = keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_path, "model_epoch_{epoch}_rmse_{RMSE}_valrmse_{val_RMSE}"), 
                                            monitor="val_RMSE", mode="min", save_weights_only=True, save_best_only=True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_RMSE', factor=0.1, mode='min', patience=1, min_lr=0.00001)
    else:
        es = keras.callbacks.EarlyStopping(monitor='val_prediction_RMSE', mode='min', patience=3, restore_best_weights=True)
        ckpt = keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_path, "model_epoch_{epoch}_rmse_{prediction_RMSE}_valrmse_{val_prediction_RMSE}"), 
                                            monitor="val_prediction_RMSE", mode="min", save_weights_only=True, save_best_only=True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_prediction_RMSE', factor=0.1, mode='min', patience=1, min_lr=0.00001)

    csv_log = keras.callbacks.CSVLogger(filename=os.path.join(log_path, 'training.log'))
    tb = keras.callbacks.TensorBoard(log_dir=tb_path)

    return [es, ckpt, csv_log, tb, reduce_lr]
