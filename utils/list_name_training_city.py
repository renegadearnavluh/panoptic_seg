import glob 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--src_root_path', type=str, default="/media/tuan/Daten/project/dataset/cityscapes/cityscapes/leftImg8bit/train",
                    help='root path of the source training dataset')
parser.add_argument('--output_name', type=str, default="train_split.txt",
                    help='name of the output file')

if __name__=="__main__":

    args = parser.parse_args()

    with open(args.output_name, 'w') as f:
        for i in glob.glob("{}/*".format(args.src_root_path)):
            print(i)
            city_name = i.split('/')[-1]
            f.write("{}\n".format(city_name))