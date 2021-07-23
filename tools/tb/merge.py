import argparse
from typing import Text
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--src1", type=str)
parser.add_argument("--src2", type=str)
parser.add_argument("--dest", type=str)
args= parser.parse_args()

summary_writer = tf.summary.FileWriter(args.dest)


def savetofile(summary_writer, src):
    for event in tf.train.summary_iterator(src):
        '''
        wall_time: 1602237006.66
        step: 18665
        summary {
        value {
            tag: "Train_loss_scale/16384.0/stable_steps"
            simple_value: 1129.0
        }
        }
        '''

        for value in event.summary.value:
                print(event.step, value.tag)
                summary = tf.Summary()
                for value in event.summary.value:
                    print(value.tag)
                    if (value.HasField('simple_value')):
                        print(value.simple_value)
                        summary.value.add(tag='{}'.format(value.tag),simple_value=value.simple_value)

                summary_writer.add_summary(summary, event.step)
                summary_writer.flush()

savetofile(summary_writer, args.src1)
savetofile(summary_writer, args.src2)
