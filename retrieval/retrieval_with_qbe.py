import argparse
import logging
import os

import torch
import torch.cuda
from   torch.utils.data                           import  DataLoader
import numpy                                      as      np
from   skimage                                    import  io
from   scipy.spatial.distance                     import  braycurtis, cosine
from   operator                                   import  itemgetter
import tqdm

from   retrieval.dataset_class.retrieval_dataset  import  RetrievalDataset
from   src.models.quaternion_phocnet_2            import  QuaternionPHOCNet
from   src.models.myphocnet                       import  PHOCNet
from   src.utils.save_load                        import  my_torch_load

import matplotlib.pyplot                          as      plt


def retrieve(retr_dataset, query_image, query_image_name, dists, results_to_show, drop_first=False):
    # Plot results by rows of 4 images
    sum_of_images = results_to_show + 1
    if sum_of_images < 3:
        nrows = 1
        ncols = sum_of_images
    else:
        nrows = sum_of_images // 3 + 1 if sum_of_images % 3 != 0 else sum_of_images // 3
        ncols = 3

    fig = plt.figure(figsize=(20,10), dpi=150)

    ax = fig.add_subplot(nrows, ncols, 1)
    ax.title.set_text('Query image')
    # Since it is grayscale, plot one of the three channels, converted to numpy
    if query_image.shape[1] > 1:
        plt.imshow(query_image[0,1,:,:].cpu().numpy(), cmap='gray')
    else:
        plt.imshow(query_image[0,0,:,:].cpu().numpy(), cmap='gray')

    # Show top 'results_to_show' images
    for i in range(results_to_show):
        idx = i + 1 if drop_first else i

        matched_word_image, matched_page_id = retr_dataset[dists[idx][0]]
        matched_word_image = matched_word_image.numpy()[0,:,:]

        ax = fig.add_subplot(nrows, ncols, i+2)

        title_str = 'Result: {}\nDistance: {:.5f}'.format(i+1, dists[idx][1])
        ax.title.set_text(title_str)

        plt.imshow(matched_word_image, cmap='gray')

    plt.tight_layout()
    plt.savefig('retrieval/retrieval_results/ranked_list_' + query_image_name + '.png')
    plt.show()


'''
    Retrieval is made either using cosine or Bray-Curtis distance.
'''
def calculate_dists(dataset_estimated_phocs, query_estimated_phoc, metric):

    dists = [(i, metric(query_estimated_phoc, dataset_estimated_phocs[i]))
                        for i in range(dataset_estimated_phocs.shape[0])]
    dists = sorted(dists, key=itemgetter(1))

    return dists


'''
    Input:  img_tensor of shape (1, 1, x, y)
    Output: quat_img_tensor of shape (1, 4, x, y) with real part = 0
'''
def make_image_quaternionic(img_tensor):
    img_tensor = img_tensor.expand(-1,3,-1,-1) if img_tensor.size(1) == 1 else img_tensor
    quat_img_tensor = torch.zeros(img_tensor.size(0), 4, img_tensor.size(2), img_tensor.size(3))
    quat_img_tensor[:, 1:, :, :] = img_tensor[:,:,:,:]

    return quat_img_tensor


def main():
    n_out_temp = 252

    logger = logging.getLogger('KWS::retrieve')
    logger.info('--- Running retrieval process ---')

    # Define arguments
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--dataset', '-ds', required=True, choices=['gw', 'iam'],
                             help='The dataset to retrieve from. Options: gw, iam')
    args_parser.add_argument('--model_type', '-mt', required=True, choices=['p', 'q'],
                             help='Type of model to use. Options: p (original PHOCNet, q (Quaternion PHOCNet))')
    args_parser.add_argument('--images_file_extension', '-ext', required=True,
                             help='Images file extension. .<extension>')
    #args_parser.add_argument('--query_image_path', '-q', required=True,
    #                         help='The path of query image.')
    args_parser.add_argument('--images_root_dir_path', '-i', required=True,
                             help='The path to the root folder where document images are located.')
    args_parser.add_argument('--results_to_show', '-r', type=int, default=5,
                             help='Number of top matched images to show. Default=5.')
    args_parser.add_argument('--phocnet_model_path', '-m', required=True,
                             help='The path of model to load.')
    args_parser.add_argument('--metric', '-metric', default='co', choices=['bc', 'co'],
                             help='The metric to use to calculate distances.')
    args_parser.add_argument('--drop_first', '-df', default=False,
                             help='Whether to drop first retrieved image or not. Default=False.')
    # Single GPU only
    args_parser.add_argument('--gpu_id', '-gpu', type=int, default=None,
                             help='The ID of GPU to use. If not specified, CPU is used.')

    args = args_parser.parse_args()

    # Print parameters
    logger.info('### Command-line parameters: ###')
    for key, value in vars(args).items():
        logger.info('%s: %s', str(key), str(value))
    logger.info('################################')

    # Check CUDA environment availability
    if args.gpu_id is not None and not torch.cuda.is_available():
        logger.warning('Could not find CUDA environment, operating in CPU mode.')
        args.gpu_id = None

    logger.info('Loading dataset to retrieve from..')
    retr_dataset = RetrievalDataset(args.dataset,
                                    args.images_root_dir_path,
                                    args.images_file_extension)
    retr_loader = DataLoader(retr_dataset)

    # Load original PHOCNet
    if args.model_type == 'p':
        logger.info('Loading original trained PHOCNet..')
        cnn = PHOCNet(n_out=n_out_temp,
                      input_channels=1,
                      gpp_type='spp',
                      pooling_levels=3)
    # Load Quaternion PHOCNet
    elif args.model_type == 'q':
        logger.info('Loading trained Quaternion-PHOCNet..')
        cnn = QuaternionPHOCNet(n_out=n_out_temp,
                                in_channels=4,
                                gpp_type='spp',
                                pooling_levels=3)

    my_torch_load(cnn, args.phocnet_model_path)

    # Print trainable parameters of model
    #model_total_params = sum(p.numel() for p in cnn.parameters() if p.requires_grad)

    #print('Number of trainable parameters: ', model_total_params)
    #exit()

    # Enable cuda for cnn model
    if args.gpu_id is not None:
        cnn.cuda(args.gpu_id)

    cnn.eval()

    dataset_estimated_phocs = np.zeros((len(retr_loader), n_out_temp))

    # Estimate phocs for retrieval dataset
    logger.info('Estimating PHOCs..')
    for idx, (word_image, page_id) in enumerate(tqdm.tqdm(retr_loader)):

        # If Quaternion phocnet, make image quaternionic
        if args.model_type == 'q':
            word_image = make_image_quaternionic(word_image)

        if args.gpu_id is not None:
            word_image = word_image.cuda(args.gpu_id)

        estimated_phoc = torch.sigmoid(cnn(word_image))
        dataset_estimated_phocs[idx] = estimated_phoc.data.cpu().numpy().flatten()

        #if idx == 100:
        #    break

    # Decide distance metric
    if args.metric == 'bc':
        metric = braycurtis
    elif args.metric == 'co':
        metric = cosine

    while True:
        logger.info('New query..')

        # Estimate phoc for query image
        if args.dataset == 'gw':
            query_image_path = input('Document image path: ')
            query_image_name = '.'.join(os.path.split(query_image_path)[1].split('.')[:-1])

            query_coords = input('Coordinates (ul_x, ul_y, lr_x, lr_y): ')
            # Convert to list of ints
            query_coords = list(map(int, query_coords.split(',')))

            query_page = io.imread(query_image_path)
            query_image = query_page[query_coords[1]:query_coords[3],
                                     query_coords[0]:query_coords[2]].copy()
        elif args.dataset == 'iam':
            query_image_path = input('Query image path: ')
            query_image_name = '.'.join(os.path.split(query_image_path)[1].split('.')[:-1])

            query_image = io.imread(query_image_path)
        else:
            raise Exception('Dataset not supported. Avaliable options: gw, iam.')

        query_image = 1 - query_image.astype(np.float32) / 255.0

        query_image = torch.from_numpy(query_image)
        query_image = query_image.reshape((1,1) + query_image.shape)

        # If quaternion phocnet, make image quaternionic
        if args.model_type == 'q':
            query_image = make_image_quaternionic(query_image)

        if args.gpu_id is not None:
            query_image = query_image.cuda(args.gpu_id)

        query_estimated_phoc = torch.sigmoid(cnn(query_image))
        query_estimated_phoc = query_estimated_phoc.data.cpu().numpy().flatten()

        # Calculate distances
        logger.info('Calculating distances..')
        dists = calculate_dists(dataset_estimated_phocs, query_estimated_phoc, metric)

        # Retrieve images
        logger.info('Retrieving..')
        retrieve(retr_dataset,
                 query_image,
                 query_image_name,
                 dists,
                 args.results_to_show,
                 args.drop_first)


if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    main()
