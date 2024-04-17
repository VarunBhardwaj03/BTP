import tensorflow as tf
import os
import glob
import numpy as np
import argparse

from train_helper import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Training options.')
parser.add_argument('-d', '--data', metavar='--data', type=str, nargs='?', help='Training data path')
args = parser.parse_args()

data_path = args.data
os.system('python3 create_tf_record.py --tfrecord_filename=mars --dataset_dir=%s' % (args.data))

# data_filename = os.path.join(args.data, 'data_summary.txt')
# with tf.io.gfile.GFile(data_filename, 'r') as f:
#     num_validatiaon = f.readline()
#     num_dataset = f.readline()

#     print('Found %d images in the training data' % (int(num_dataset) - int(num_validatiaon)))
#     print('Found %d images in the validataion data' % (int(num_validatiaon)))
#     training_data_num = int(num_dataset) - int(num_validatiaon)

training_data_num = 0
num_validatiaon = 0
for root_dir, cur_dir, files in os.walk(os.path.join(args.data, 'bbox_train')):
    training_data_num += len(files)

for root_dir, cur_dir, files in os.walk(os.path.join(args.data, 'bbox_test')):
    num_validatiaon += len(files)

num_dataset = num_validatiaon+training_data_num

print('Found %d images in the training data' % (int(num_dataset) - int(num_validatiaon)))
print('Found %d images in the validataion data' % (int(num_validatiaon)))



if __name__ == "__main__":

    BATCH_SIZE = 32
    num_epochs = 200
    tf.compat.v1.disable_eager_execution()
    # train_dataset = combine_dataset(batch_size=BATCH_SIZE, image_size=[256, 128], same_prob=0.95, diff_prob=.001, train=True)
    # val_dataset = combine_dataset(batch_size=BATCH_SIZE, image_size=[256, 128], same_prob=0.95, diff_prob=.001, train=False)
    train_dataset = combine_dataset(batch_size=BATCH_SIZE, image_size=[256, 128], same_prob=0.5, diff_prob=0.5, train=True)
    val_dataset = combine_dataset(batch_size=BATCH_SIZE, image_size=[256, 128], same_prob=0.5, diff_prob=0.5, train=False)
    print(type(train_dataset))
    handle = tf.compat.v1.placeholder(tf.string, shape=[])

    # train_iterator = train_dataset.make_one_shot_iterator()
    # val_iterator = val_dataset.make_one_shot_iterator()
    # train_iterator = iter(train_dataset)
    # val_iterator = iter(val_dataset)
    train_iterator = tf.compat.v1.data.make_one_shot_iterator(train_dataset)
    val_iterator = tf.compat.v1.data.make_one_shot_iterator(val_dataset)    
    iterator = tf.compat.v1.data.Iterator.from_string_handle(
                handle, train_iterator.output_types, train_iterator.output_shapes)

    

    left, right = iterator.get_next()
    left_input_im, left_label, left_addr = left
    right_input_im, right_label, right_addr = right

    logits, model_left, model_right = inference(left_input_im, right_input_im)
    loss(logits, left_label, right_label)
    contrastive_loss(model_left, model_right, logits, left_label, right_label, margin=0.2, use_loss=True)
    total_loss = tf.compat.v1.losses.get_total_loss()
    global_step = tf.Variable(0, trainable=False)

    params = tf.compat.v1.trainable_variables()
    gradients = tf.gradients(total_loss, params)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
    updates = optimizer.apply_gradients(zip(gradients, params), global_step=global_step)


    global_init = tf.compat.v1.variables_initializer(tf.compat.v1.global_variables())

    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(global_init)

        # setup tensorboard
        if not os.path.exists('train.log'):
            os.makedirs('train.log')
        tf.summary.scalar('step', global_step)
        tf.summary.scalar('loss', total_loss)
        for var in tf.compat.v1.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        merged = tf.compat.v1.summary.merge_all()
        writer = tf.compat.v1.summary.FileWriter('train.log', sess.graph)
        training_handle = sess.run(train_iterator.string_handle())

        validation_handle = sess.run(val_iterator.string_handle())

        num_iterations = training_data_num // BATCH_SIZE

        for epoch in range(num_epochs):
            print('epoch : ', epoch, ' / ', num_epochs)
            for iteration in range(num_iterations):
                # print("hi")
                # print(num_iterations)
                feed_dict_train = {handle:training_handle}
                # print(feed_dict_train)
                # print(total_loss)
                # print(updates)
                # print(merged)
                loss_train, _ = sess.run([total_loss, updates], feed_dict_train)
                # writer.add_summary(summary_str, epoch)
                print("iteration : %d / %d - Loss : %f" % (iteration, num_iterations, loss_train))

            feed_dict_val = {handle: validation_handle}
            val_loss = sess.run([total_loss], feed_dict_val)
            print('========================================')
            print("epoch : %d - Validation Loss : %f" % (epoch, val_loss[0]))
            print('========================================')

            if not os.path.exists("./model_siamese/"):
                os.makedirs("./model_siamese/")
            saver.save(sess, "model_siamese/model.ckpt")