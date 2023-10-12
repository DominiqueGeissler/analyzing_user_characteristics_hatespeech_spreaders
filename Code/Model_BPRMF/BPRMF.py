'''
Code adapted from "Causal Understanding of Fake News Dissemination on Social Media", by Cheng et. al in KDD 2021
'''
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import pandas as pd
from Code.utility.helper import *
from Code.utility.batch_test import *
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from logger import get_logger


logger = get_logger(__name__ + args.model_type)

class BPRMF(object):
    def __init__(self, data_config, p_score=None):
        self.model_type = args.model_type

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.lr = args.lr
        # self.lr_decay = args.lr_decay
        self.clip = 0.01


        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)

        self.regs = eval(args.regs)
        self.decay = self.regs[0]

        self.verbose = args.verbose

        # placeholder definition
        self.users = tf.compat.v1.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.compat.v1.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.compat.v1.placeholder(tf.int32, shape=(None,))

        # pscore definition only needed for unbiased embeddings.
        if p_score is not None:
            self.p_score = tf.constant(p_score, dtype=tf.float32)
            self.p_score = tf.clip_by_value(
                tf.constant(p_score, dtype=tf.float32), clip_value_min=self.clip, clip_value_max=1.0)
            # propensity-score
            self.pscore_pos = tf.gather(self.p_score, self.pos_items)

        self.weights = self._init_weights()

        # Original embedding.
        u_e = tf.nn.embedding_lookup(params=self.weights['user_embedding'], ids=self.users)
        pos_i_e = tf.nn.embedding_lookup(params=self.weights['item_embedding'], ids=self.pos_items)
        neg_i_e = tf.nn.embedding_lookup(params=self.weights['item_embedding'], ids=self.neg_items)

        # All ratings for all users.
        self.batch_ratings = tf.matmul(u_e, pos_i_e, transpose_a=False, transpose_b=True)

        # Compute loss.
        self.mf_loss, self.reg_loss = self.create_bpr_loss(u_e, pos_i_e, neg_i_e)
        self.loss = self.mf_loss + self.reg_loss
        self.training_loss = []
        self.opt = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.loss)

        self._statistics_params()

    def _init_weights(self):
        # initialize weights for users and hatespeech items.
        # used as (un)biased embeddings.
        all_weights = dict()

        initializer = tf.compat.v1.initializers.glorot_uniform()

        all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
        all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')

        return all_weights

    def create_bpr_loss(self, users, pos_items, neg_items):
        # compute loss for each batch
        pos_scores = tf.reduce_sum(input_tensor=tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(input_tensor=tf.multiply(users, neg_items), axis=1)

        # regularization
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size
        reg_loss = self.decay * regularizer

        if self.model_type == "bprmf_V" or self.model_type == "bprmf_F":
            # unbiased loss
            maxi = tf.maximum(tf.negative(tf.math.log(tf.nn.sigmoid((pos_scores - neg_scores)))) / self.pscore_pos, 0)
            mf_loss = tf.reduce_mean(input_tensor=maxi)
        else:
            # biased loss
            maxi = tf.math.log(tf.nn.sigmoid(pos_scores - neg_scores))
            mf_loss = tf.negative(tf.reduce_mean(input_tensor=maxi))

        return mf_loss, reg_loss

    def _statistics_params(self):
        # number of params
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        if self.verbose > 0:
            logger.debug("#params: %d" % total_parameters)

    def train(self, sess):
        # training the model
        # loop through epochs
        logger.info('start training the model...')
        for epoch in range(args.epoch):
            t1 = time()
            loss, mf_loss, reg_loss = 0., 0., 0.
            # n_batch = data_generator.n_train // args.batch_size + 1
            dataset = data_generator.exist_users
            rd.shuffle(dataset)

            # divide the dataset into batches
            for batch in range(0, len(dataset) - args.batch_size + 1, args.batch_size):
                users = dataset[batch:batch + args.batch_size]
                pos_items, neg_items = data_generator.sample(users)
                # training
                _, batch_loss, batch_mf_loss, batch_reg_loss = sess.run(
                    [model.opt, model.loss, model.mf_loss, model.reg_loss],
                    feed_dict={model.users: users, model.pos_items: pos_items,
                               model.neg_items: neg_items})

                loss += batch_loss
                mf_loss += batch_mf_loss
                reg_loss += batch_reg_loss

            if np.isnan(loss) == True:
                logger.error('ERROR: loss is nan.')
                sys.exit()

            # save the training loss
            model.training_loss.append((epoch, loss))
            # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
            if (epoch) % 10 == 0:
                if args.verbose > 0 and epoch % args.verbose == 0:
                    perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, reg_loss)
                    logger.debug(perf_str)
                continue


if __name__ == '__main__':
    # *********************************************************
    logger.info("Starting model: " + args.model_type)
    # set up performance measures
    recall = np.zeros((args.n_repetitions, len(Ks)))
    precision = np.zeros((args.n_repetitions, len(Ks)))
    ndcg = np.zeros((args.n_repetitions, len(Ks)))

    data_generator.print_statistics()

    # repeat the experiment for n_repetitions
    for i in range(args.n_repetitions):
        logger.debug(f'This is repetition {i} of {args.n_repetitions} repetitions.')

        config = dict()
        config['n_users'] = data_generator.n_users
        config['n_items'] = data_generator.n_items
        logger.debug(f"Nr of users: {config['n_users']}, nr of items: {config['n_items']}")

        # initialize the model
        if args.model_type == 'bprmf_t':
            lambda_ls = 1e-2
            args.regs = '[' + str(lambda_ls) + ',1e-6,1e-8]'
            pscore = np.load(args.data_path + '/pscore_t.npy')
            model = BPRMF(data_config=config, p_score=pscore)
            logger.debug(f"lambda_ls: {lambda_ls}; regularization: {args.regs}")
        elif args.model_type == 'bprmf_ut':
            lambda_ls = 1e-2
            args.regs = '[' + str(lambda_ls) + ',1e-6,1e-6]'
            pscore = np.load(args.data_path + '/pscore_ut.npy')
            model = BPRMF(data_config=config, p_score=pscore)
            logger.debug(f"lambda_ls: {lambda_ls}; regularization: {args.regs}")
        elif args.model_type == 'bprmf':
            model = BPRMF(data_config=config)
            logger.debug(f"regularization: {args.regs}")

        else:
            logger.error(f'ERROR: model type {args.model_type} is not supported.')
            sys.exit()

        t0 = time()
        saver = tf.compat.v1.train.Saver(max_to_keep=args.n_repetitions)

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)

        # *********************************************************
        # reload the pretrained model parameters.
        if args.pretrain == 1:
            pretrain_path = '../%sweights/%s/l%s_r%s_rep-%s' % (args.proj_path, model.model_type, str(args.lr),
                                                                '-'.join([str(r) for r in eval(args.regs)]), str(i))
            # ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path + '/checkpoint'))
            if os.path.exists(pretrain_path+'.index'):
                print('load the pretrained model parameters from: ', pretrain_path)
                sess.run(tf.compat.v1.global_variables_initializer())
                saver.restore(sess, pretrain_path)

            else:
                logger.debug('no pretrained model parameters to load from: ', pretrain_path)
                sess.run(tf.compat.v1.global_variables_initializer())
                cur_best_pre_0 = 0.
                logger.info('Train from scratch.')
                model.train(sess)

                # *********************************************************
                # save the model parameters.
                if args.save_flag == 1:
                    weights_save_path = '../%sweights/%s/l%s_r%s_rep' % (args.proj_path, model.model_type, str(args.lr),
                                                                         '-'.join([str(r) for r in eval(args.regs)]))
                    ensureDir(weights_save_path)
                    print('save the model parameters to: ', weights_save_path + '-' + str(i))
                    saver.save(sess, weights_save_path, global_step=i)
        else:
            sess.run(tf.compat.v1.global_variables_initializer())
            cur_best_pre_0 = 0.
            logger.info('Train the model without pretraining.')
            model.train(sess)

            # *********************************************************
            # save the model parameters.
            if args.save_flag == 1:
                weights_save_path = '../%sweights/%s/l%s_r%s_rep' % (args.proj_path, model.model_type, str(args.lr),
                                                                 '-'.join([str(r) for r in eval(args.regs)]))
                ensureDir(weights_save_path)
                print('save the model parameters to: ', str(weights_save_path) + '-' + str(i))
                saver.save(sess, weights_save_path, global_step=i)


        # plt.plot(*zip(*model.training_loss))
        # plt.show()

        # *********************************************************
        # test the performance of the model.
        logger.info('start testing the model...')
        users_to_test = list(data_generator.test_set.keys())
        logger.debug(f"Nr of users to test: {len(users_to_test)}")
        ret = test(sess, model, users_to_test, drop_flag=False)
        logger.debug("Recall: %s", str(ret['recall']))
        logger.debug("Precision: %s", str(ret['precision']))
        logger.debug("NDCG: %s", str(ret['ndcg']))
        recall[i, :] = ret['recall']
        precision[i, :] = ret['precision']
        ndcg[i, :] = ret['ndcg']

        if i == 0:
            user_embedding = ret['user_embedding']
        else:
            user_embedding += ret['user_embedding']
        logger.debug("shape of the user embeddings: %s", str(ret['user_embedding'].shape))
        tf.compat.v1.reset_default_graph()

    # save model performance
    performance = {'model': args.model_type,
                   'n_repetitions': args.n_repetitions,
                   'epochs': args.epoch,
                   'batch_size': args.batch_size,
                   'lr': args.lr,
                   'Ks': Ks,
                   'recall_avg': np.mean(recall, axis=0),
                   'precision_avg': np.mean(precision, axis=0),
                   'ndcg_avg': np.mean(ndcg, axis=0),
                   'recall_run1': recall[0, :],
                   'precision_run1': precision[0, :],
                   'ndcg_run1': ndcg[0, :],
                   'recall_run2': recall[1, :],
                   'precision_run2': precision[1, :],
                   'ndcg_run2': ndcg[1, :],
                   'recall_run3': recall[2, :],
                   'precision_run3': precision[2, :],
                   'ndcg_run3': ndcg[2, :],
                   'recall_run4': recall[3, :],
                   'precision_run4': precision[3, :],
                   'ndcg_run4': ndcg[3, :],
                   'recall_run5': recall[4, :],
                   'precision_run5': precision[4, :],
                   'ndcg_run5': ndcg[4, :]
                   }
    df = pd.DataFrame.from_dict(performance)
    performance_path = '../results/bprmf/%s_rep%s_e%s_bs%s_' % \
                       (args.model_type, args.n_repetitions, args.epoch, args.batch_size)
    df.to_csv(performance_path + 'performance.csv', index=False)
    logger.debug("Average Recall: %s", str(np.mean(recall, axis=0)))
    logger.debug("Average Precision: %s", str(np.mean(precision, axis=0)))
    logger.debug("Average NDCG: %s", str(np.mean(ndcg, axis=0)))

    # save user embedding
    logger.info('save user embedding...')
    embedding_path = '../results/bprmf/%s_' % args.model_type
    np.save(embedding_path + 'user_embedding.npy', arr=user_embedding / 5)



