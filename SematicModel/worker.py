import argparse
import tensorflow as tf
import time
import random
import os
import dot_product_similarity_model
import importlib

parser = argparse.ArgumentParser()
parser.add_argument('--task', default='email', choices=['email'])
parser.add_argument('--model', default='dot-product', choices=['dot-product','jointly'])
parser.add_argument('--mode', default='train', choices=['train', 'eval'])
parser.add_argument('--checkpoint-frequency', type=int, default=100)
parser.add_argument('--eval-frequency', type=int, default=200)
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument("--device", default="/cpu:0")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--embedding_size", type=int, default=320)
parser.add_argument("--epochs", type=int, default=10)
args = parser.parse_args()


task_name = args.task

task = importlib.import_module(task_name)

checkpoint_dir = os.path.join(task.train_dir, 'checkpoint')
tflog_dir = os.path.join(task.train_dir, 'tflog')
checkpoint_name = task_name + '-model'
checkpoint_dir = os.path.join(task.train_dir, 'checkpoints')
checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

n_grams_size = task.ngrams_size()
print "n_grmas size length: %s" % n_grams_size


def SematicSimilarityModel(session, restore_only=False):
    is_training = tf.placeholder(dtype=tf.bool, name='is_training')

    model = dot_product_similarity_model(
        n_grams_size=n_grams_size,
        embedding_size=args.embedding_size,
        batch_size=args.batch_size,
        is_training=is_training,
        learning_rate=args.lr,
        device=args.device
    )

    saver = tf.train.Saver(tf.global_variables())
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    if checkpoint:
        print("Reading model parameters from %s \n" % checkpoint.model_checkpoint_path)
        saver.restore(session, checkpoint.model_checkpoint_path)
    elif restore_only:
        print("Cannot restore model")
    else:
        print("Created model with fresh parameters")
        session.run(tf.global_variables_initializer())
    # tf.get_default_graph().finalize()
    return model, saver


model_fn = SematicSimilarityModel


def batch_iterator(dataset, batch_size, max_epochs):
    for i in range(max_epochs):
        xb = []
        yb = []
        random.shuffle(dataset)
        for ex in dataset:
            x, y = ex
            xb.append(x)
            yb.append(y)
            if len(xb) == batch_size:
                yield xb, yb
                xb, yb = [], []


def dev_iterator(dataset):
    xb = []
    yb = []
    for ex in dataset:
        x, y = ex
        xb.append(x)
        yb.append(y)
    return (xb, yb)


def train():
    tf.reset_default_graph()
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        model, saver = model_fn(sess)
        train_summary_dir = os.path.join(tflog_dir,"train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, graph=tf.get_default_graph())

        dev_summary_dir = os.path.join(tflog_dir,"dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, graph=tf.get_default_graph())

        global_step = model.global_step

        def train_step(x, y):
            fd = model.get_feed_data(x, y)
            t0 = time.clock()
            step, x_inputs_sum_embedded, y_inputs_sum_embedded, similarity, loss,summaries, _ = sess.run([
                model.global_step,
                model.final_x_inputs_sum_embedded,
                model.final_y_inputs_sum_embedded,
                model.similarity,
                model.loss,
                model.summary_op,
                model.train_op,
            ], fd)
            td = time.clock() - t0
            print('step %s, loss=%s, t=%s \n' % (step, loss, round(td, 5)))
            print('similarity_matrix is  \n')
            print(similarity)

            print (x_inputs_sum_embedded)
            print (y_inputs_sum_embedded)
            train_summary_writer.add_summary(summaries, global_step=step)

        def dev_step(x, y):
            fd = model.get_feed_data(x, y)
            step, summaries, loss = sess.run([
                model.global_step,
                model.summary_op,
                model.loss
            ], fd)

            print('evaluation at step %s \n' % step)
            #print('dev accuracy: %.5f \n' % accuracy)
            dev_summary_writer.add_summary(summaries, global_step=step)

        devset = task.read_devset(epochs=1)
        dev_x,dev_y = dev_iterator(devset)
        for i, (x, y) in enumerate(batch_iterator(task.read_trainset(epochs=args.epochs), args.batch_size, 100)):
            train_step(x, y)
            current_step = tf.train.global_step(sess, global_step)
            if current_step != 0 and current_step % args.checkpoint_frequency == 0:
                print('checkpoint & graph meta \n')
                saver.save(sess, checkpoint_path, global_step=current_step)
                print('checkpoint done \n')
            if current_step != 0 and current_step % args.eval_frequency == 0:
                dev_step(dev_x,dev_y)

def main():
    if args.mode == 'train':
        train()
    elif args.mode == 'eval':
        pass
    #evaluate(task.read_testset(epochs=1))

if __name__ == '__main__':
  main()
