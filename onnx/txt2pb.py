import onnx
import argparse
from typing import Text

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str)

args= parser.parse_args()

model_proto = onnx.load(args.src)
#onnx.save(model_proto, args.src + ".txt",format=Text)


text_file = open(args.src + ".pb", "wb")
text_file.write(model_proto.SerializeToString())
text_file.close()
