import tensorflow as tf
import numpy as np
import cv2
import os

import image_util as iu
import logging
import sys
import yaml
import subprocess
import my_util

class ImageDataSet:

    CONFIG_YAML = 'nail_config.yml'

    OUT_DIR = 'outdir/'
    CROPPED_DIR = 'cropped/'
    CASCADE_XML_DIR = 'hog/'
    MY_ANNOTATION_DIR = 'my_annotation/'
    MY_ANNOTATION_IMG_DIR = 'bbox/'

    def __init__(self):

        # log setting
        program = os.path.basename(sys.argv[0])
        self.logger = logging.getLogger(program)
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')

        # load config file
        f = open(self.CONFIG_YAML, 'r')
        self.config = yaml.load(f)
        f.close()

        self.__init_dir()

        self.cascade = None

    def __init_dir(self):

        # assign empty value if dictionary is not set
        if 'dataset' not in self.config:
            self.config['dataset'] = None
        if 'pos_img_dir' not in self.config['dataset']:
            self.config['dataset']['pos_img_dir'] = ''
        if 'neg_img_dir' not in self.config['dataset']:
            self.config['dataset']['neg_img_dir'] = ''
        if 'test_img_dir' not in self.config['dataset']:
            self.config['dataset']['test_img_dir'] = ''
        if 'output' not in self.config:
            self.config['output'] = None
        if 'output_dir' not in self.config['output']:
            self.config['dataset']['output_dir'] = ''

        # set dataset path
        self.pos_img_dir = self.config['dataset']['pos_img_dir']
        self.neg_img_dir = self.config['dataset']['neg_img_dir']
        self.test_img_dir = self.config['dataset']['test_img_dir']

        # set output path
        self.output_dir = self.config['output']['output_dir']
        self.out_dir = self.output_dir + self.OUT_DIR
        
        # create output paths
        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)

        # set array of all file names
        self.pos_img_files = [file_name for file_name in os.listdir(self.pos_img_dir) if not file_name.startswith('.')]
        self.pos_img_files.sort()
        self.neg_img_files = [file_name for file_name in os.listdir(self.neg_img_dir) if not file_name.startswith('.')]
        self.neg_img_files.sort()
        self.test_img_files = [file_name for file_name in os.listdir(self.test_img_dir) if not file_name.startswith('.')]
        self.test_img_files.sort()

    def save_config(self):

        # save config.yml
        self.logger.info("saving config file")
        f = open(self.CONFIG_YAML, 'w')
        f.write(yaml.dump(self.config, default_flow_style=False))
        f.close()

        # reload valuables related directories
        self.__init_dir()

    def create_positive_dat_with_my_annotation(self):
        output_text = ""
        self.logger.info("begin creating positive.dat")
        for file_name in self.my_annotation_files:

            # annotation path
            annotation_path = self.my_annotation_dir + file_name
            bboxes = my_util.my_unpickle(annotation_path)
            root, ext = os.path.splitext(file_name)
            output_text += "%s  %d  " % (self.pos_img_dir + root, len(bboxes))
            for bbox in bboxes:
                x_min, y_min = min(bbox[0][0], bbox[1][0]), min(bbox[0][1], bbox[1][1])
                x_max, y_max = max(bbox[0][0], bbox[1][0]), max(bbox[0][1], bbox[1][1])
                w = x_max - x_min
                h = y_max - y_min
                output_text += "%d %d %d %d  " % (x_min, y_min, w, h)
            output_text += "\n"

        self.logger.info("writing data to positive.dat")
        f = open('positive.dat', 'w')
        f.write(output_text)
        f.close()
        self.logger.info("completed writing data to positive.dat")

    def create_positive_dat_by_image_size(self):
        output_text = ""
        self.logger.info("begin creating positive.dat")
        for file_name in self.pos_img_files:

            file_path = self.pos_img_dir + file_name
            im = cv2.imread(file_path)
            output_text += "%s  %d  " % (file_path, 1)
            output_text += "%d %d %d %d  \n" % (0, 0, im.shape[0], im.shape[1])
        self.logger.info("writing data to positive.dat")
        f = open('positive.dat', 'w')
        f.write(output_text)
        f.close()
        self.logger.info("completed writing data to positive.dat")

    def create_samples(self, use_my_annotation=False, width=24, height=24):

        if use_my_annotation:
            self.create_positive_dat_with_my_annotation()
        else:
            self.create_positive_dat_by_image_size()
        self.create_negative_dat()

        params = {
            'info': 'positive.dat',
            'vec': 'positive.vec',
            'num': len(self.pos_img_files),
            'width': width,
            'height': height
        }
        cmd = "opencv_createsamples -info %(info)s -vec %(vec)s -num %(num)d -w %(width)d -h %(height)d" % params
        self.logger.info("running command: %s", cmd)
        subprocess.call(cmd.strip().split(" "))

    def create_negative_dat(self):
        output_text = ""
        self.logger.info("begin creating negative.dat")
        for file_name in self.neg_img_files:

            file_path = self.neg_img_dir + file_name
            output_text += file_path
            output_text += "\n"
        self.logger.info("writing data to negative.dat")
        f = open('negative.dat', 'w')
        f.write(output_text)
        f.close()
        self.logger.info("completed writing data to negative.dat")

    def inside(self, r, q):
        rx, ry, rw, rh = r
        qx, qy, qw, qh = q
        return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

    def get_annotation_existence_list(self):
        return [file + '.pkl' in self.my_annotation_files for file in self.pos_img_files]

    def get_annotated_image_files(self):
        return [os.path.splitext(annotation_file)[0] for annotation_file in self.my_annotation_files]

    def get_annotation_path(self, img_file):
        annotation_path = self.my_annotation_dir + img_file + '.pkl'
        return annotation_path

    def get_img_file_by_annotation_file(self, annotation_file):
        img_file = os.path.splitext(annotation_file)[0]
        return img_file

    def read_img(self, img_file):
        self.logger.info('loading image file: %s', img_file)
        img_path = self.pos_img_dir + img_file

        # read image
        cv_img = cv2.imread(img_path)
        return cv_img

if __name__ == '__main__':

    logging.root.setLevel(level=logging.INFO)

    dataset = ImageDataSet()
