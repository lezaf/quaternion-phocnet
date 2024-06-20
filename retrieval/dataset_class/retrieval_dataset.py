import  os

import  torch
from    torch.utils.data                import  Dataset
from    skimage                         import  io
import  numpy                           as      np

from    src.transformations.image_size  import check_size


class RetrievalDataset(Dataset):
    def __init__(self, dataset, images_root_dir_path, image_file_extension='.tif'):
        '''
            Args:
                dataset (string):              Which dataset to load. Supported: gw, iam
                images_root_dir_path (string): Path to the images root directory.
                image_file_extension (string): File extension of images to load.
            Notes:
                words: list with tuples in format (word_image, page_id)
        '''

        self.words = []

        # words_seg: tuples (word_image, page_id)
        if dataset == 'gw':
            self.words = self._segment_images_to_words(images_root_dir_path, image_file_extension)
        elif dataset == 'iam':
            self.words = self._load_segmented_words(images_root_dir_path, image_file_extension)
        else:
            raise Exception('Dataset not supported. Avaliable options: gw, iam.')

        #for word_seg in words_seg:
        #    word_image = word_seg[0]

            # Scale pixels in range [0,1]: 0 -> white, 1 -> black
        #    word_image = 1 - word_image.astype(np.float32) / 255.0

        #    self.words.append((word_image, word_seg[1]))


    def __len__(self):
        return len(self.words)


    def __getitem__(self, idx):
        word_image = self.words[idx][0]

        word_image = word_image.reshape((1,) + word_image.shape)
        word_image = torch.from_numpy(word_image)

        page_id = self.words[idx][1]

        return word_image, page_id


    '''
        Current version of function uses annotation file to segment the image into words.

        Args:
            images_root_dir_path (string): Root directory should include following directories:
                * 'pages': images with the pages to retrieve from
                * 'ground_truth': .gtp files with annotations in the form 'ul_x ul_y lr_x lr_y transcr'

        Returns:
                list with tuples: (word_image, page_id)
    '''
    @staticmethod
    def _segment_images_to_words(images_root_dir_path, image_file_extension):

        image_filenames = [file for file in os.listdir(os.path.join(images_root_dir_path, 'pages'))
                                if file.endswith(image_file_extension)]

        words_seg = []

        for image_filename in image_filenames:
            page_id = '.'.join(image_filename.split('.')[:-1])
            doc_image = io.imread(os.path.join(images_root_dir_path, 'pages', image_filename))

            # Filename with the ground truth for words
            gt_filename = '.'.join(image_filename.split('.')[:-1] + ['gtp'])

            with open(os.path.join(images_root_dir_path, 'ground_truth', gt_filename)) as f:
                gt_lines = f.readlines()

            # Remove white spaces and '\n'
            gt_lines = [line.strip() for line in gt_lines]

            # Segment words
            for line in gt_lines:
                ul_x, ul_y, lr_x, lr_y = list(map(int, line.split(' ')[:-1]))
                word_image = doc_image[ul_y:lr_y, ul_x:lr_x].copy()

                # Scale pixels in range [0,1]: 0 -> white, 1 -> black
                word_image = 1 - word_image.astype(np.float32) / 255.0

                words_seg.append((word_image, page_id))

        return words_seg


    '''
        Notes:
            * min_image_width_height: hardcoded
    '''
    @staticmethod
    def _load_segmented_words(images_root_dir_path, image_file_extension):
        words_seg = []

        words_dir = os.path.join(images_root_dir_path, 'words')
        for dir_name in os.listdir(words_dir):
            for sub_dir_name in os.listdir(os.path.join(words_dir, dir_name)):
                for image_filename in os.listdir(os.path.join(words_dir, dir_name, sub_dir_name)):

                    image_id = '.'.join(image_filename.split('.')[:-1])
                    try:
                        word_image = io.imread(os.path.join(words_dir, dir_name, sub_dir_name, image_filename))
                    except:
                        continue

                    # Scale pixels in range [0,1]: 0 -> white, 1 -> black
                    word_image = 1 - word_image.astype(np.float32) / 255.0
                    word_image = check_size(img=word_image, min_image_width_height=30)

                    words_seg.append((word_image, image_id))

        return words_seg