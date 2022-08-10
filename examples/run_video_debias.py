from absl import app
from absl import flags

from sklearn.metrics import mean_absolute_error, mean_squared_error

import numpy as np

import tensorflow as tf
from tensorflow import keras

import os
import sys
sys.path.append('../')

from deepctr.layers.utils import metrics_at_k
from deepctr.feature_column import get_feature_names
from examples.utils import get_feature_target_duration_mean_std, load_yaml, get_data, split_data, get_model, get_callbacks

from datetime import datetime

FLAGS = flags.FLAGS
flags.DEFINE_string('name', 'debug', 'Exp name.')
flags.DEFINE_string('model', 'DeepFM', 'Recommendation model.')
flags.DEFINE_boolean('run_eagerly', False, 'Whether to debug with eager mode.')
flags.DEFINE_boolean('test_ckpt', False, 'Whether to load checkpoint and evaluation.')
flags.DEFINE_string('save_path', './', 'Save path if test_ckpt is True.')
flags.DEFINE_boolean('save_result', True, 'Whether to save the recommendation results.')
flags.DEFINE_enum('dataset', 'kuaishou', ['kuaishou', 'wechat'], 'Dataset.')
flags.DEFINE_enum('train_target', 'watchtime', ['watchtime', 'gain'], 'Use which target to train.')
flags.DEFINE_boolean('remove_duration_feature', False, 'Whether to remove duration from input features.')
flags.DEFINE_boolean('post_transform', False, 'Whether to transform the predicted watchtime to gain.')
flags.DEFINE_integer('embedding_dim', 4, 'Embedding size.')
flags.DEFINE_integer('dnn_hidden_units', 32, 'Hidden units of DNN.')
flags.DEFINE_boolean('disentangle', False, 'Whether to debias with disentanglement learning.')
flags.DEFINE_float('disentangle_loss_weight', 1.0, 'Loss weight for disentanglement learning.')
flags.DEFINE_integer('gpu_id', 1, 'GPU ID.')
flags.DEFINE_integer('epochs', 100, 'The number of epochs.')
flags.DEFINE_integer('batch_size', 512, 'Batch size.')

flags.register_validator('name',
    lambda value: value != 'debug',
    message='Exp name not informative! Provide a valid exp name!')


def train_model(flags_obj, cfg, save_path, model, feature_names, data_splits, model_input, target, duration):

    callbacks = get_callbacks(flags_obj, save_path)
    train_model_input = model_input[0]
    train = data_splits[0]
    val_model_input = model_input[1]
    val = data_splits[1]

    if not flags_obj.disentangle:

        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.MeanSquaredError(name='mse_loss'),
            metrics=[keras.metrics.MeanAbsoluteError(name='MAE'), keras.metrics.RootMeanSquaredError(name='RMSE')],
            run_eagerly=flags_obj.run_eagerly)

        train_label = train[target].values
        validation_label = val[target].values

    else:

        prediction_output_name = 'prediction'

        independent_output_name = 'independent_reconstruct'
        loss = keras.losses.MeanSquaredError(name='mse_loss')
        loss_weights = {
            prediction_output_name: 1.0,
            independent_output_name: flags_obj.disentangle_loss_weight
        }
        metrics = [keras.metrics.MeanAbsoluteError(name='MAE'), keras.metrics.RootMeanSquaredError(name='RMSE')]

        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=loss,
            metrics=metrics,
            loss_weights=loss_weights,
            run_eagerly=flags_obj.run_eagerly)

        train_label = {
            prediction_output_name: train[target].values,
            independent_output_name: train[duration].values.astype(float)
        }
        validation_label = {
            prediction_output_name: val[target].values,
            independent_output_name: val[duration].values.astype(float)
        }

    history = model.fit(
        x=train_model_input,
        y=train_label,
        batch_size=flags_obj.batch_size,
        epochs=flags_obj.epochs,
        verbose=2,
        validation_data=(val_model_input, validation_label),
        callbacks=callbacks)

    return history


def test_model(flags_obj, cfg, save_path, model, data_splits, model_input, target, watchtime_target, gain_target, like, mean_play, std_play):

    uid_feature_name = cfg['data'][flags_obj.dataset]['uid_feature_name']
    duration_feature_name = cfg['data'][flags_obj.dataset]['duration_feature_name']
    iid_feature_name = cfg['data'][flags_obj.dataset]['iid_feature_name']
    (_, _, test) = data_splits
    (_, _, test_model_input) = model_input
    if flags_obj.model not in ['Random', 'Long', 'Short', 'Optimum']:
        if not flags_obj.disentangle:
            pred_ans = model.predict(test_model_input, batch_size=flags_obj.batch_size)
        else:
            pred_ans, _ = model.predict(test_model_input, batch_size=flags_obj.batch_size)
    elif flags_obj.model != 'Optimum':
        pred_ans = model.predict(test[duration_feature_name].values)
    else:
        pred_ans = model.predict(test[target].values)

    if flags_obj.post_transform:
        mean_play_value = test[mean_play].values
        std_play_value = test[std_play].values
        pred_ans = (pred_ans - mean_play_value)/std_play_value

    MAE = round(mean_absolute_error(test[target].values.reshape(-1), pred_ans.reshape(-1)), 4)
    RMSE = round(mean_squared_error(test[target].values.reshape(-1), pred_ans.reshape(-1), squared=False), 4)
    MAE_AT_K, RMSE_AT_K, WATCHTIME_AT_K, GAIN_AT_K, DCG_AT_K, LIKE_AT_K, BC_AT_K = metrics_at_k(
        test[target].values.reshape(-1), 
        test[watchtime_target].values.reshape(-1),
        test[gain_target].values.reshape(-1),
        test[like].values.reshape(-1),
        pred_ans.reshape(-1), 
        test[uid_feature_name].values.reshape(-1), 
        test[duration_feature_name].values.reshape(-1),
        cfg['test'][flags_obj.dataset]['top_k'],
        4,
        flags_obj.save_result,
        os.path.join(save_path, cfg['train']['save_result_path']),
        flags_obj.name,
        flags_obj.dataset,
        flags_obj.model,
        flags_obj.train_target,
        flags_obj.disentangle,
        test[iid_feature_name].values.reshape(-1),
        cfg['data'][flags_obj.dataset]['bc_thres'])
    print("test MAE", MAE)
    print("test RMSE", RMSE)
    print("test MAE@K", MAE_AT_K)
    print("test RMSE@K", RMSE_AT_K)
    print("test WATCHTIME@K", WATCHTIME_AT_K)
    print("test GAIN@K", GAIN_AT_K)
    print("test DCG@K", DCG_AT_K)
    print("test LIKE@K", LIKE_AT_K)
    print("test BC@K", BC_AT_K)

    result = {
        'MAE': MAE,
        'RMSE': RMSE,
        'MAE@K': MAE_AT_K,
        'RMSE@K': RMSE_AT_K,
        'WATCHTIME@K': WATCHTIME_AT_K,
        'GAIN@K': GAIN_AT_K,
        'DCG@K': DCG_AT_K,
        'LIKE@K': LIKE_AT_K,
        'BC@K': BC_AT_K
    }

    return result


def main(argv):

    flags_obj = FLAGS

    os.environ["CUDA_VISIBLE_DEVICES"] = str(flags_obj.gpu_id)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'

    cfg = load_yaml('./config.yaml')

    #read data
    data = get_data(flags_obj, cfg)

    #transform data into feature columns
    data, fixlen_feature_columns, watchtime_target, gain_target, like, duration, mean_play, std_play = get_feature_target_duration_mean_std(flags_obj, cfg, data)
    if flags_obj.train_target == 'watchtime':
        target = watchtime_target
    elif flags_obj.train_target == 'gain':
        target = gain_target
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    #generate input data for model
    data_splits, model_input = split_data(flags_obj, cfg, data, feature_names)

    #define Model,train,predict and evaluate
    model = get_model(flags_obj, cfg, linear_feature_columns, dnn_feature_columns)

    if not flags_obj.test_ckpt:
        save_path = os.path.join(cfg['train']['save_path'], 
                                 flags_obj.dataset + '-' + flags_obj.model + '-' + datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(save_path)
        model_plot_path = os.path.join(save_path, 'model.png')
        keras.utils.plot_model(model, model_plot_path, show_shapes=True)
        print('Model plot at: {}'.format(model_plot_path))
        history = train_model(flags_obj, cfg, save_path, model, feature_names, data_splits, model_input, target, duration)

        result = test_model(flags_obj, cfg, save_path, model, data_splits, model_input, target, watchtime_target, gain_target, like, mean_play, std_play)
        latest = tf.train.latest_checkpoint(os.path.join(save_path, 'ckpt'))
        print('Latest checkpoint at: {}'.format(latest))
    else:
        if flags_obj.model not in ['Random', 'Long', 'Short', 'Optimum']:
            latest = tf.train.latest_checkpoint(os.path.join(flags_obj.save_path, 'ckpt'))
            print('Loading latest checkpoint at: {}'.format(latest))
            model.load_weights(latest).expect_partial()
        else:
            latest = 'None'
        result = test_model(flags_obj, cfg, flags_obj.save_path, model, data_splits, model_input, target, watchtime_target, gain_target, like, mean_play, std_play)

    if flags_obj.model not in ['Random', 'Long', 'Short', 'Optimum']:
        train_rmse, val_rmse = map(float, latest.split('_rmse_')[1].split('_valrmse_'))
    else:
        train_rmse, val_rmse = 0.0, 0.0
    result['Train RMSE'] = round(train_rmse, 4)
    result['Val RMSE'] = round(val_rmse, 4)


if __name__ == "__main__":

    app.run(main)
